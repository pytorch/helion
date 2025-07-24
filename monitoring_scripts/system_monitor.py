#!/usr/bin/env python3
"""System resource monitoring script for multi-GPU contention analysis"""

import psutil
import time
import json
from datetime import datetime
import subprocess
import signal
import os
import platform

class SystemMonitor:
    def __init__(self, output_file, sample_interval=0.1):
        self.output_file = output_file
        self.sample_interval = sample_interval
        self.running = True
        self.results = {
            'start_time': datetime.now().isoformat(),
            'sample_interval_ms': sample_interval * 1000,
            'system_info': self._get_system_info(),
            'samples': []
        }
        
        # Track Python processes
        self.python_pids = set()
        
    def _get_system_info(self):
        """Collect static system information"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'numa_nodes': self._get_numa_info()
        }
        return info
    
    def _get_numa_info(self):
        """Get NUMA topology information"""
        try:
            # Try to get NUMA info from lscpu
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            numa_info = {}
            
            for line in lines:
                if 'NUMA node' in line and 'CPU(s):' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        node_name = parts[0].strip()
                        cpus = parts[1].strip()
                        numa_info[node_name] = cpus
            
            return numa_info
        except:
            return None
    
    def _get_numa_stats(self):
        """Get current NUMA statistics"""
        try:
            result = subprocess.run(['numastat', '-c'], capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                return result.stdout
        except:
            pass
        return None
    
    def _get_pcie_info(self):
        """Get PCIe topology information"""
        pcie_info = {}
        try:
            # Get GPU PCIe info from nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=gpu_bus_id,pcie.link.gen.current,pcie.link.width.current', 
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) == 3:
                        pcie_info[f'gpu_{i}'] = {
                            'bus_id': parts[0],
                            'pcie_gen': parts[1],
                            'pcie_width': parts[2]
                        }
        except:
            pass
        return pcie_info
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.running = False
    
    def collect_sample(self, start_time):
        """Collect a single sample of system metrics"""
        sample = {
            'timestamp': time.time() - start_time,
            'cpu': {},
            'memory': {},
            'processes': {},
            'io': {}
        }
        
        # CPU metrics
        sample['cpu']['percent_per_core'] = psutil.cpu_percent(interval=0, percpu=True)
        sample['cpu']['percent_total'] = sum(sample['cpu']['percent_per_core']) / len(sample['cpu']['percent_per_core'])
        
        # CPU frequency
        cpu_freq = psutil.cpu_freq(percpu=True)
        if cpu_freq:
            sample['cpu']['freq_per_core'] = [{'current': f.current, 'min': f.min, 'max': f.max} for f in cpu_freq]
        
        # Context switches and interrupts
        cpu_stats = psutil.cpu_stats()
        sample['cpu']['ctx_switches'] = cpu_stats.ctx_switches
        sample['cpu']['interrupts'] = cpu_stats.interrupts
        
        # Memory metrics
        vm = psutil.virtual_memory()
        sample['memory']['total_gb'] = vm.total / (1024**3)
        sample['memory']['used_gb'] = vm.used / (1024**3)
        sample['memory']['available_gb'] = vm.available / (1024**3)
        sample['memory']['percent'] = vm.percent
        sample['memory']['buffers_gb'] = vm.buffers / (1024**3) if hasattr(vm, 'buffers') else None
        sample['memory']['cached_gb'] = vm.cached / (1024**3) if hasattr(vm, 'cached') else None
        
        # Swap memory
        swap = psutil.swap_memory()
        sample['memory']['swap_used_gb'] = swap.used / (1024**3)
        sample['memory']['swap_percent'] = swap.percent
        
        # NUMA statistics
        numa_stats = self._get_numa_stats()
        if numa_stats:
            sample['memory']['numa_stats'] = numa_stats
        
        # Process information
        python_processes = []
        total_python_cpu = 0
        total_python_mem = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
            try:
                info = proc.info
                if 'python' in info['name'].lower():
                    # Track new Python processes
                    if info['pid'] not in self.python_pids:
                        self.python_pids.add(info['pid'])
                    
                    # Get more detailed info for Python processes
                    proc_data = {
                        'pid': info['pid'],
                        'cpu_percent': info['cpu_percent'],
                        'memory_percent': info['memory_percent'],
                        'cmdline': ' '.join(info['cmdline']) if info['cmdline'] else ''
                    }
                    
                    # Try to get CPU affinity
                    try:
                        proc_data['cpu_affinity'] = proc.cpu_affinity()
                    except:
                        proc_data['cpu_affinity'] = None
                    
                    # Try to get memory info
                    try:
                        mem_info = proc.memory_info()
                        proc_data['rss_mb'] = mem_info.rss / (1024**2)
                        proc_data['vms_mb'] = mem_info.vms / (1024**2)
                    except:
                        pass
                    
                    python_processes.append(proc_data)
                    total_python_cpu += info['cpu_percent'] or 0
                    total_python_mem += info['memory_percent'] or 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        sample['processes']['python_processes'] = python_processes
        sample['processes']['python_count'] = len(python_processes)
        sample['processes']['python_total_cpu_percent'] = total_python_cpu
        sample['processes']['python_total_mem_percent'] = total_python_mem
        
        # IO statistics
        io_counters = psutil.disk_io_counters()
        if io_counters:
            sample['io']['read_mb'] = io_counters.read_bytes / (1024**2)
            sample['io']['write_mb'] = io_counters.write_bytes / (1024**2)
            sample['io']['read_count'] = io_counters.read_count
            sample['io']['write_count'] = io_counters.write_count
        
        # Network IO (might be relevant for distributed training)
        net_io = psutil.net_io_counters()
        if net_io:
            sample['io']['net_sent_mb'] = net_io.bytes_sent / (1024**2)
            sample['io']['net_recv_mb'] = net_io.bytes_recv / (1024**2)
        
        return sample
    
    def monitor(self, duration=None):
        """Main monitoring loop"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        start_time = time.time()
        
        print(f"Starting system monitoring...")
        print(f"Output file: {self.output_file}")
        print(f"Sample interval: {self.sample_interval * 1000:.0f}ms")
        if duration:
            print(f"Duration: {duration}s")
        print("Press Ctrl+C to stop...")
        
        # Get initial PCIe info
        self.results['pcie_topology'] = self._get_pcie_info()
        
        while self.running:
            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                break
            
            # Collect sample
            sample = self.collect_sample(start_time)
            self.results['samples'].append(sample)
            
            # Sleep until next sample
            time.sleep(self.sample_interval)
        
        # Cleanup
        self.save_results()
        print(f"\nMonitoring complete. Results saved to {self.output_file}")
    
    def save_results(self):
        """Save results to JSON file"""
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_samples'] = len(self.results['samples'])
        self.results['duration_seconds'] = self.results['samples'][-1]['timestamp'] if self.results['samples'] else 0
        self.results['total_python_processes_seen'] = len(self.python_pids)
        
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor system resources for contention analysis')
    parser.add_argument('output_file', help='Output JSON file for monitoring data')
    parser.add_argument('--duration', type=float, help='Monitoring duration in seconds (default: unlimited)')
    parser.add_argument('--interval', type=float, default=0.1, help='Sample interval in seconds (default: 0.1)')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.output_file, args.interval)
    monitor.monitor(args.duration)

if __name__ == '__main__':
    main()