#!/usr/bin/env python3
"""Analyze variance in auto-tuning results between baseline and concurrent runs"""

import json
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path

class VarianceAnalyzer:
    def __init__(self):
        self.results = {
            'baseline': defaultdict(dict),
            'concurrent': defaultdict(dict),
            'isolated': defaultdict(dict)
        }
        
    def load_results(self, files, category):
        """Load benchmark results from files"""
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract kernel name and GPU info from filename
                filename = Path(file_path).stem
                parts = filename.split('_')
                
                if 'gpu' in filename:
                    # Extract GPU number
                    gpu_idx = None
                    for part in parts:
                        if 'gpu' in part:
                            gpu_idx = part.replace('gpu', '')
                            break
                    
                    kernel_name = data.get('kernel_name', 'unknown')
                    
                    self.results[category][kernel_name][gpu_idx] = {
                        'best_config': data.get('best_config', {}),
                        'best_time': data.get('best_time_ms', None),
                        'all_times': data.get('all_benchmark_times', []),
                        'num_configs_tested': data.get('num_configs_tested', 0),
                        'autotune_duration': data.get('autotune_duration_seconds', 0),
                        'file': file_path
                    }
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def analyze_timing_variance(self, times):
        """Calculate statistical measures for timing data"""
        if not times:
            return {}
        
        times = np.array(times)
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'cv': np.std(times) / np.mean(times) if np.mean(times) > 0 else 0,  # Coefficient of variation
            'min': np.min(times),
            'max': np.max(times),
            'range': np.max(times) - np.min(times),
            'percentiles': {
                '25': np.percentile(times, 25),
                '50': np.percentile(times, 50),
                '75': np.percentile(times, 75),
                '95': np.percentile(times, 95)
            }
        }
    
    def compare_configs(self, config1, config2):
        """Compare two configurations and return differences"""
        if not config1 or not config2:
            return None
        
        differences = {}
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                differences[key] = {'baseline': val1, 'concurrent': val2}
        
        return differences
    
    def analyze_results(self):
        """Perform comprehensive variance analysis"""
        print("=" * 80)
        print("AUTO-TUNING VARIANCE ANALYSIS")
        print("=" * 80)
        
        # Analyze each kernel
        all_kernels = set()
        for category in self.results:
            all_kernels.update(self.results[category].keys())
        
        for kernel in sorted(all_kernels):
            print(f"\n\nKERNEL: {kernel}")
            print("-" * 60)
            
            # Get baseline results
            baseline_data = self.results['baseline'].get(kernel, {})
            concurrent_data = self.results['concurrent'].get(kernel, {})
            isolated_data = self.results['isolated'].get(kernel, {})
            
            # Compare baseline vs concurrent for each GPU
            for gpu_id in sorted(set(baseline_data.keys()) | set(concurrent_data.keys())):
                print(f"\n  GPU {gpu_id}:")
                
                base = baseline_data.get(gpu_id, {})
                conc = concurrent_data.get(gpu_id, {})
                isol = isolated_data.get(gpu_id, {})
                
                # Best time comparison
                if base.get('best_time') and conc.get('best_time'):
                    base_time = base['best_time']
                    conc_time = conc['best_time']
                    diff_pct = ((conc_time - base_time) / base_time) * 100
                    
                    print(f"    Best Time:")
                    print(f"      Baseline:   {base_time:.3f} ms")
                    print(f"      Concurrent: {conc_time:.3f} ms")
                    print(f"      Difference: {diff_pct:+.1f}%")
                    
                    if isol.get('best_time'):
                        isol_time = isol['best_time']
                        isol_diff = ((isol_time - base_time) / base_time) * 100
                        print(f"      Isolated:   {isol_time:.3f} ms ({isol_diff:+.1f}% vs baseline)")
                
                # Configuration comparison
                if base.get('best_config') and conc.get('best_config'):
                    config_diff = self.compare_configs(base['best_config'], conc['best_config'])
                    if config_diff:
                        print(f"    Configuration Differences:")
                        for param, values in config_diff.items():
                            print(f"      {param}: {values['baseline']} → {values['concurrent']}")
                    else:
                        print(f"    Configuration: SAME")
                
                # Timing variance analysis
                if base.get('all_times') and conc.get('all_times'):
                    base_stats = self.analyze_timing_variance(base['all_times'])
                    conc_stats = self.analyze_timing_variance(conc['all_times'])
                    
                    print(f"    Timing Variance:")
                    print(f"      Baseline:   CV={base_stats['cv']:.3f}, σ={base_stats['std']:.3f}ms")
                    print(f"      Concurrent: CV={conc_stats['cv']:.3f}, σ={conc_stats['std']:.3f}ms")
                    
                    # Calculate variance increase
                    var_increase = (conc_stats['std'] / base_stats['std'] - 1) * 100 if base_stats['std'] > 0 else 0
                    print(f"      Variance increase: {var_increase:+.1f}%")
                
                # Auto-tuning efficiency
                if base.get('autotune_duration') and conc.get('autotune_duration'):
                    print(f"    Auto-tuning Duration:")
                    print(f"      Baseline:   {base['autotune_duration']:.1f}s")
                    print(f"      Concurrent: {conc['autotune_duration']:.1f}s")
            
            # Cross-GPU analysis
            if len(concurrent_data) > 1:
                print(f"\n  Cross-GPU Analysis:")
                
                # Check if different GPUs found different optimal configs
                configs = [gpu_data['best_config'] for gpu_data in concurrent_data.values() 
                          if gpu_data.get('best_config')]
                
                if configs:
                    # Simple check: are all configs the same?
                    config_strs = [json.dumps(c, sort_keys=True) for c in configs]
                    unique_configs = len(set(config_strs))
                    
                    if unique_configs > 1:
                        print(f"    WARNING: {unique_configs} different optimal configurations found across GPUs!")
                    else:
                        print(f"    All GPUs found the same optimal configuration")
                
                # Best time variance across GPUs
                best_times = [gpu_data['best_time'] for gpu_data in concurrent_data.values() 
                             if gpu_data.get('best_time')]
                
                if best_times:
                    time_stats = self.analyze_timing_variance(best_times)
                    print(f"    Best time variance across GPUs: CV={time_stats['cv']:.3f}")
        
        # Summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total_comparisons = 0
        config_mismatches = 0
        significant_slowdowns = 0
        variance_increases = 0
        
        for kernel in self.results['baseline']:
            for gpu_id in self.results['baseline'][kernel]:
                if gpu_id in self.results['concurrent'].get(kernel, {}):
                    total_comparisons += 1
                    
                    base = self.results['baseline'][kernel][gpu_id]
                    conc = self.results['concurrent'][kernel][gpu_id]
                    
                    # Check config mismatch
                    if base.get('best_config') != conc.get('best_config'):
                        config_mismatches += 1
                    
                    # Check slowdown
                    if base.get('best_time') and conc.get('best_time'):
                        diff_pct = ((conc['best_time'] - base['best_time']) / base['best_time']) * 100
                        if diff_pct > 5:  # More than 5% slowdown
                            significant_slowdowns += 1
                    
                    # Check variance increase
                    if base.get('all_times') and conc.get('all_times'):
                        base_std = np.std(base['all_times'])
                        conc_std = np.std(conc['all_times'])
                        if base_std > 0 and (conc_std / base_std) > 1.5:  # 50% increase
                            variance_increases += 1
        
        print(f"Total comparisons: {total_comparisons}")
        print(f"Configuration mismatches: {config_mismatches} ({config_mismatches/total_comparisons*100:.1f}%)")
        print(f"Significant slowdowns (>5%): {significant_slowdowns} ({significant_slowdowns/total_comparisons*100:.1f}%)")
        print(f"Variance increases (>50%): {variance_increases} ({variance_increases/total_comparisons*100:.1f}%)")
        
        if config_mismatches > 0 or significant_slowdowns > total_comparisons * 0.2:
            print("\n⚠️  CONCLUSION: Significant resource contention detected!")
            print("   Running benchmarks concurrently on different GPUs causes:")
            print("   - Different optimal configurations to be selected")
            print("   - Increased timing variance")
            print("   - Suboptimal auto-tuning results")
        else:
            print("\n✓  CONCLUSION: Minimal resource contention detected.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_variance.py <baseline_files> [concurrent_files] [isolated_files]")
        print("  Files should be comma-separated lists")
        print("  Example: python analyze_variance.py baseline1.json,baseline2.json concurrent1.json,concurrent2.json")
        sys.exit(1)
    
    analyzer = VarianceAnalyzer()
    
    # Load baseline files
    baseline_files = sys.argv[1].split(',')
    analyzer.load_results(baseline_files, 'baseline')
    
    # Load concurrent files if provided
    if len(sys.argv) > 2:
        concurrent_files = sys.argv[2].split(',')
        analyzer.load_results(concurrent_files, 'concurrent')
    
    # Load isolated files if provided
    if len(sys.argv) > 3:
        isolated_files = sys.argv[3].split(',')
        analyzer.load_results(isolated_files, 'isolated')
    
    # Analyze results
    analyzer.analyze_results()

if __name__ == '__main__':
    main()