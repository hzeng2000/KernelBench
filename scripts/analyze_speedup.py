#!/usr/bin/env python3
"""
分析 KernelBench 评测结果的脚本
计算每个case的加速比（对比多个baseline），以及汇总运行通过比例和平均加速比
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import math


# Baseline文件名到显示名称的映射
BASELINE_FILES = {
    'torch': 'baseline_time_torch.json',
    'compile_cudagraphs': 'baseline_time_torch_compile_cudagraphs.json',
    'compile_default': 'baseline_time_torch_compile_inductor_default.json',
    'compile_max_autotune_no_cg': 'baseline_time_torch_compile_inductor_max-autotune-no-cudagraphs.json',
    'compile_max_autotune': 'baseline_time_torch_compile_inductor_max-autotune.json',
    'compile_reduce_overhead': 'baseline_time_torch_compile_inductor_reduce-overhead.json',
}

# 简短显示名称
BASELINE_SHORT_NAMES = {
    'torch': 'Eager',
    'compile_cudagraphs': 'CUDAGraphs',
    'compile_default': 'Default',
    'compile_max_autotune_no_cg': 'MaxAT-noCG',
    'compile_max_autotune': 'MaxAT',
    'compile_reduce_overhead': 'ReduceOH',
}


def load_json(file_path: Path) -> Dict:
    """加载JSON文件"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_all_baselines(baseline_dir: Path, level: str) -> Dict[str, Dict]:
    """
    加载所有baseline数据
    
    Returns:
        Dict[baseline_name, Dict[problem_id, baseline_time]]
    """
    baselines = {}
    
    for name, filename in BASELINE_FILES.items():
        filepath = baseline_dir / filename
        if filepath.exists():
            data = load_json(filepath)
            if level in data:
                level_data = data[level]
                # 构建problem_id到时间的映射
                id_to_time = {}
                id_to_filename = {}
                for prob_filename, timing in level_data.items():
                    parts = prob_filename.split('_')
                    if parts[0].isdigit():
                        problem_id = parts[0]
                        id_to_time[problem_id] = timing.get('mean')
                        id_to_filename[problem_id] = prob_filename
                baselines[name] = {
                    'times': id_to_time,
                    'filenames': id_to_filename
                }
        else:
            print(f"Warning: Baseline file not found: {filepath}", file=sys.stderr)
    
    return baselines


def analyze_results(
    eval_results_path: Path,
    baseline_dir: Path,
    level: str = "level2",
) -> Dict[str, Any]:
    """
    分析评测结果，对比所有baseline
    
    Args:
        eval_results_path: 评测结果JSON文件路径
        baseline_dir: baseline时间数据目录路径  
        level: 评测级别 (level1, level2, level3, level4)
    
    Returns:
        分析结果字典
    """
    # 加载数据
    eval_results = load_json(eval_results_path)
    baselines = load_all_baselines(baseline_dir, level)
    
    if not baselines:
        print(f"Error: No baseline data found for {level}", file=sys.stderr)
        return {}
    
    # 分析结果
    results = []
    total_cases = 0
    passed_cases = 0
    
    # 每个baseline的加速比列表
    speedups_by_baseline = {name: [] for name in baselines.keys()}
    
    for problem_id, samples in eval_results.items():
        total_cases += 1
        
        # 获取第一个sample的结果 (通常只有一个sample)
        sample = samples[0] if samples else None
        if sample is None:
            continue
        
        # 检查是否通过 (compiled=True 且 correctness=True)
        compiled = sample.get('compiled', False)
        correctness = sample.get('correctness', False)
        passed = compiled and correctness
        
        # 获取运行时间
        runtime = sample.get('runtime', -1.0)
        if runtime == -1.0:
            runtime = None
        
        if passed:
            passed_cases += 1
        
        # 获取错误信息
        error_info = None
        if not passed:
            metadata = sample.get('metadata', {})
            if 'runtime_error' in metadata:
                error_info = metadata.get('runtime_error_name', 'RuntimeError')
            elif 'other_error' in metadata:
                error_info = metadata.get('other_error_name', 'OtherError')
            elif not compiled:
                error_info = "CompileError"
            elif not correctness:
                error_info = "IncorrectResult"
        
        # 计算每个baseline的加速比
        baseline_times = {}
        speedups = {}
        baseline_filename = None
        
        for bl_name, bl_data in baselines.items():
            bl_time = bl_data['times'].get(problem_id)
            baseline_times[bl_name] = bl_time
            if baseline_filename is None:
                baseline_filename = bl_data['filenames'].get(problem_id)
            
            if passed and runtime is not None and bl_time is not None and runtime > 0:
                speedup = bl_time / runtime
                speedups[bl_name] = speedup
                speedups_by_baseline[bl_name].append(speedup)
            else:
                speedups[bl_name] = None
        
        results.append({
            'problem_id': problem_id,
            'baseline_file': baseline_filename,
            'compiled': compiled,
            'correctness': correctness,
            'passed': passed,
            'runtime': runtime,
            'baseline_times': baseline_times,
            'speedups': speedups,
            'error': error_info
        })
    
    # 排序结果 (按problem_id数字排序)
    results.sort(key=lambda x: int(x['problem_id']))
    
    # 计算汇总统计
    pass_rate = passed_cases / total_cases * 100 if total_cases > 0 else 0
    
    # 每个baseline的统计
    baseline_stats = {}
    for bl_name, sp_list in speedups_by_baseline.items():
        if sp_list:
            log_sum = sum(math.log(s) for s in sp_list)
            geomean = math.exp(log_sum / len(sp_list))
            baseline_stats[bl_name] = {
                'count': len(sp_list),
                'avg': sum(sp_list) / len(sp_list),
                'geomean': geomean,
                'min': min(sp_list),
                'max': max(sp_list),
            }
        else:
            baseline_stats[bl_name] = {
                'count': 0,
                'avg': None,
                'geomean': None,
                'min': None,
                'max': None,
            }
    
    summary = {
        'total_cases': total_cases,
        'passed_cases': passed_cases,
        'pass_rate': pass_rate,
        'baseline_stats': baseline_stats,
    }
    
    return {
        'results': results,
        'summary': summary,
        'baseline_names': list(baselines.keys())
    }


def get_case_name(baseline_file: str) -> str:
    """从baseline文件名提取case名称"""
    if not baseline_file:
        return "-"
    # 移除.py后缀和开头的数字ID
    name = baseline_file.replace('.py', '')
    parts = name.split('_', 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1]
    return name


def print_table(analysis: Dict[str, Any], show_all: bool = False, show_baseline_times: bool = False):
    """打印表格格式的结果"""
    results = analysis['results']
    summary = analysis['summary']
    baseline_names = analysis['baseline_names']
    
    # 打印表头
    print("\n" + "=" * 180)
    print("KernelBench 评测结果分析")
    print("=" * 180)
    
    # 如果需要显示baseline时间，先打印baseline时间表
    if show_baseline_times:
        print("\n【Baseline 运行时间 (ms)】")
        bl_time_headers = " | ".join([f"{BASELINE_SHORT_NAMES.get(n, n):>10}" for n in baseline_names])
        time_header = f"{'ID':>4} | {'Case Name':<40} | {bl_time_headers}"
        print(time_header)
        print("-" * len(time_header))
        
        for r in results:
            case_name = get_case_name(r['baseline_file'])
            if len(case_name) > 40:
                case_name = case_name[:37] + "..."
            
            bl_time_strs = []
            for bl_name in baseline_names:
                bl_time = r['baseline_times'].get(bl_name)
                if bl_time is not None:
                    bl_time_strs.append(f"{bl_time:.2f}")
                else:
                    bl_time_strs.append("-")
            
            bl_time_values = " | ".join([f"{s:>10}" for s in bl_time_strs])
            print(f"{r['problem_id']:>4} | {case_name:<40} | {bl_time_values}")
        print()
    
    # 打印详细结果表格（加速比）
    print("\n【加速比结果】")
    # 构建表头
    bl_headers = " | ".join([f"{BASELINE_SHORT_NAMES.get(n, n):>10}" for n in baseline_names])
    header = f"{'ID':>4} | {'Case Name':<40} | {'Pass':>5} | {'Runtime':>10} | {bl_headers}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        pass_str = "✓" if r['passed'] else "❌"
        runtime_str = f"{r['runtime']:.2f}" if r['runtime'] else "-"
        case_name = get_case_name(r['baseline_file'])
        if len(case_name) > 40:
            case_name = case_name[:37] + "..."
        
        # 每个baseline的加速比
        speedup_strs = []
        for bl_name in baseline_names:
            sp = r['speedups'].get(bl_name)
            if sp is not None:
                speedup_strs.append(f"{sp:.2f}x")
            elif r['error']:
                speedup_strs.append(r['error'][:10])
            else:
                speedup_strs.append("-")
        
        bl_values = " | ".join([f"{s:>10}" for s in speedup_strs])
        print(f"{r['problem_id']:>4} | {case_name:<40} | {pass_str:>5} | {runtime_str:>10} | {bl_values}")
    
    # 打印汇总
    print("\n" + "=" * 120)
    print("汇总统计")
    print("=" * 120)
    print(f"总Case数:        {summary['total_cases']}")
    print(f"通过Case数:      {summary['passed_cases']}")
    print(f"通过率:          {summary['pass_rate']:.2f}%")
    
    print("\n各Baseline加速比统计:")
    print("-" * 80)
    print(f"{'Baseline':<25} | {'Count':>6} | {'Avg':>10} | {'GeoMean':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 80)
    
    for bl_name in baseline_names:
        stats = summary['baseline_stats'].get(bl_name, {})
        display_name = BASELINE_SHORT_NAMES.get(bl_name, bl_name)
        count = stats.get('count', 0)
        avg = f"{stats['avg']:.2f}x" if stats.get('avg') else "-"
        geomean = f"{stats['geomean']:.2f}x" if stats.get('geomean') else "-"
        min_sp = f"{stats['min']:.2f}x" if stats.get('min') else "-"
        max_sp = f"{stats['max']:.2f}x" if stats.get('max') else "-"
        
        print(f"{display_name:<25} | {count:>6} | {avg:>10} | {geomean:>10} | {min_sp:>10} | {max_sp:>10}")
    
    print("=" * 120)


def print_csv(analysis: Dict[str, Any]):
    """打印CSV格式的结果"""
    results = analysis['results']
    summary = analysis['summary']
    baseline_names = analysis['baseline_names']
    
    # CSV 表头
    bl_headers = ",".join([f"speedup_{n}" for n in baseline_names])
    bl_time_headers = ",".join([f"baseline_{n}_ms" for n in baseline_names])
    print(f"problem_id,case_name,baseline_file,compiled,correctness,passed,runtime_ms,{bl_time_headers},{bl_headers},error")
    
    for r in results:
        case_name = get_case_name(r['baseline_file'])
        bl_times = ",".join([str(r['baseline_times'].get(n, '')) for n in baseline_names])
        bl_speedups = ",".join([str(r['speedups'].get(n, '')) if r['speedups'].get(n) else '' for n in baseline_names])
        
        print(f"{r['problem_id']},{case_name},{r['baseline_file'] or ''},{r['compiled']},{r['correctness']},{r['passed']},"
              f"{r['runtime'] if r['runtime'] else ''},{bl_times},{bl_speedups},{r['error'] or ''}")
    
    # 汇总信息
    print(f"\n# Summary")
    print(f"# Total Cases: {summary['total_cases']}")
    print(f"# Passed Cases: {summary['passed_cases']}")
    print(f"# Pass Rate: {summary['pass_rate']:.2f}%")
    
    for bl_name in baseline_names:
        stats = summary['baseline_stats'].get(bl_name, {})
        print(f"# {bl_name} - Avg Speedup: {stats.get('avg', 'N/A')}, GeoMean: {stats.get('geomean', 'N/A')}")


def print_json(analysis: Dict[str, Any]):
    """打印JSON格式的结果"""
    print(json.dumps(analysis, indent=2))


def print_markdown(analysis: Dict[str, Any], show_all: bool = False):
    """打印Markdown格式的结果"""
    results = analysis['results']
    summary = analysis['summary']
    baseline_names = analysis['baseline_names']
    
    print("# KernelBench 评测结果分析\n")
    
    # 汇总统计
    print("## 汇总统计\n")
    print(f"| 指标 | 值 |")
    print(f"|------|-----|")
    print(f"| 总Case数 | {summary['total_cases']} |")
    print(f"| 通过Case数 | {summary['passed_cases']} |")
    print(f"| 通过率 | {summary['pass_rate']:.2f}% |")
    
    print("\n## 各Baseline加速比统计\n")
    print("| Baseline | Count | Avg | GeoMean | Min | Max |")
    print("|----------|------:|----:|--------:|----:|----:|")
    
    for bl_name in baseline_names:
        stats = summary['baseline_stats'].get(bl_name, {})
        display_name = BASELINE_SHORT_NAMES.get(bl_name, bl_name)
        count = stats.get('count', 0)
        avg = f"{stats['avg']:.2f}x" if stats.get('avg') else "-"
        geomean = f"{stats['geomean']:.2f}x" if stats.get('geomean') else "-"
        min_sp = f"{stats['min']:.2f}x" if stats.get('min') else "-"
        max_sp = f"{stats['max']:.2f}x" if stats.get('max') else "-"
        print(f"| {display_name} | {count} | {avg} | {geomean} | {min_sp} | {max_sp} |")
    
    print("\n## Baseline 运行时间 (ms)\n")
    
    # Baseline时间表头
    bl_time_headers = " | ".join([BASELINE_SHORT_NAMES.get(n, n) for n in baseline_names])
    print(f"| ID | Case Name | {bl_time_headers} |")
    bl_time_seps = " | ".join(["-------:" for _ in baseline_names])
    print(f"|----:|-----------|{bl_time_seps} |")
    
    for r in results:
        case_name = get_case_name(r['baseline_file'])
        if len(case_name) > 35:
            case_name = case_name[:32] + "..."
        
        bl_time_strs = []
        for bl_name in baseline_names:
            bl_time = r['baseline_times'].get(bl_name)
            if bl_time is not None:
                bl_time_strs.append(f"{bl_time:.2f}")
            else:
                bl_time_strs.append("-")
        
        bl_time_values = " | ".join(bl_time_strs)
        print(f"| {r['problem_id']} | {case_name} | {bl_time_values} |")
    
    print("\n## 加速比详细结果\n")
    
    # 表头
    bl_headers = " | ".join([BASELINE_SHORT_NAMES.get(n, n) for n in baseline_names])
    print(f"| ID | Case Name | Pass | Runtime (ms) | {bl_headers} |")
    
    # 分隔符
    bl_seps = " | ".join(["-------:" for _ in baseline_names])
    print(f"|----:|-----------|:----:|-------------:| {bl_seps} |")
    
    for r in results:
        pass_str = "✓" if r['passed'] else "❌"
        runtime_str = f"{r['runtime']:.2f}" if r['runtime'] else "-"
        case_name = get_case_name(r['baseline_file'])
        if len(case_name) > 35:
            case_name = case_name[:32] + "..."
        
        speedup_strs = []
        for bl_name in baseline_names:
            sp = r['speedups'].get(bl_name)
            if sp is not None:
                speedup_strs.append(f"{sp:.2f}x")
            elif r['error']:
                err_short = r['error'][:12] + "..." if len(r['error'] or '') > 12 else r['error']
                speedup_strs.append(f"_{err_short}_")
            else:
                speedup_strs.append("-")
        
        bl_values = " | ".join(speedup_strs)
        print(f"| {r['problem_id']} | {case_name} | {pass_str} | {runtime_str} | {bl_values} |")


def main():
    parser = argparse.ArgumentParser(
        description='分析 KernelBench 评测结果，计算加速比和通过率（对比多个baseline）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python analyze_speedup.py \\
    --eval runs/level_2_backend_tilelang_model_grok-code-fast-1/eval_results.json \\
    --baseline-dir results/timing/A100-SXM4-80GB \\
    --level level2

  python analyze_speedup.py \\
    --eval runs/level_2_backend_tilelang_model_grok-code-fast-1/eval_results.json \\
    --baseline-dir results/timing/A100-SXM4-80GB \\
    --format markdown > report.md

Baseline类型说明:
  - Eager:       torch eager模式 (无编译)
  - CUDAGraphs:  torch.compile with cudagraphs backend
  - Default:     torch.compile inductor default模式
  - MaxAT-noCG:  torch.compile inductor max-autotune (无cudagraphs)
  - MaxAT:       torch.compile inductor max-autotune
  - ReduceOH:    torch.compile inductor reduce-overhead模式
        """
    )
    
    parser.add_argument(
        '--eval', '-e',
        type=str,
        default="runs/level_2_backend_tilelang_model_grok-code-fast-1/eval_results.json",
        help='评测结果JSON文件路径'
    )
    
    parser.add_argument(
        '--baseline-dir', '-b', 
        type=str,
        default='results/timing/A100-SXM4-80GB',
        help='Baseline时间数据目录路径 (包含多个baseline JSON文件)'
    )
    
    parser.add_argument(
        '--level', '-l',
        type=str,
        default='level2',
        choices=['level1', 'level2', 'level3', 'level4'],
        help='评测级别 (默认: level2)'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='table',
        choices=['table', 'json', 'csv', 'markdown'],
        help='输出格式 (默认: table)'
    )
    
    parser.add_argument(
        '--show-all', '-a',
        action='store_true',
        help='显示所有详细信息（包括失败case的详情）'
    )
    
    parser.add_argument(
        '--show-baseline-times', '-t',
        action='store_true',
        help='显示各baseline的运行时间'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    eval_path = Path(args.eval)
    baseline_dir = Path(args.baseline_dir)
    
    # 检查文件是否存在
    if not eval_path.exists():
        print(f"Error: 评测结果文件不存在: {eval_path}", file=sys.stderr)
        sys.exit(1)
    
    if not baseline_dir.exists():
        print(f"Error: Baseline目录不存在: {baseline_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 分析结果
    analysis = analyze_results(eval_path, baseline_dir, args.level)
    
    if not analysis:
        print("Error: 分析失败", file=sys.stderr)
        sys.exit(1)
    
    # 输出结果
    if args.format == 'table':
        print_table(analysis, args.show_all, args.show_baseline_times)
    elif args.format == 'csv':
        print_csv(analysis)
    elif args.format == 'json':
        print_json(analysis)
    elif args.format == 'markdown':
        print_markdown(analysis, args.show_all)


if __name__ == '__main__':
    main()
