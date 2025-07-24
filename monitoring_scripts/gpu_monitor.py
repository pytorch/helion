#!/usr/bin/env python3
"""GPU resource monitoring script for multi-GPU contention analysis"""

import nvidia_ml_py as nvml
import time
import json
from datetime import datetime
import sys
import signal
import threading

class GPUMonitor:
    def __init__(self, output_file, sample_interval=0.1):
        self.output_file = output_file
        self.sample_interval = sample_interval
        self.running = True
        self.results = {
            'start_time': datetime.now().isoformat(),
            'sample_interval_ms': sample_interval * 1000,
            'samples': []
        }
        
        # Initialize NVML
        nvml.nvmlInit()
        self.device_count = nvml.nvmlDeviceGetCount()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
        
        # Get GPU info
        self.gpu_info = []
        for i, handle in enumerate(self.handles):
            info = {
                'gpu_id': i,
                'name': nvml.nvmlDeviceGetName(handle).decode('utf-8'),
                'driver_version': nvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                'pcie_link_gen': nvml.nvmlDeviceGetCurrPcieLinkGeneration(handle),
                'pcie_link_width': nvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            }
            self.gpu_info.append(info)
        
        self.results['gpu_info'] = self.gpu_info
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.running = False
        
    def collect_sample(self, start_time):
        """Collect a single sample of GPU metrics"""
        sample = {
            'timestamp': time.time() - start_time,
            'gpus': []
        }
        
        for i, handle in enumerate(self.handles):
            try:
                # Get utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get power
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to W
                except nvml.NVMLError:
                    power = None
                
                # Get temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                # Get PCIe throughput (may not be available on all GPUs)
                try:
                    tx_bytes = nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES)
                    rx_bytes = nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES)
                except nvml.NVMLError:
                    tx_bytes = rx_bytes = None
                
                # Get clock speeds
                sm_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                
                # Get running processes
                try:
                    processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    process_count = len(processes)
                    process_mem_mb = sum(p.usedGpuMemory / (1024**2) for p in processes)
                except nvml.NVMLError:
                    process_count = 0
                    process_mem_mb = 0
                
                gpu_data = {
                    'gpu_id': i,
                    'gpu_util': util.gpu,
                    'mem_util': util.memory,
                    'mem_used_gb': mem_info.used / (1024**3),
                    'mem_total_gb': mem_info.total / (1024**3),
                    'power_w': power,
                    'temp_c': temp,
                    'sm_clock_mhz': sm_clock,
                    'mem_clock_mhz': mem_clock,
                    'pcie_tx_mb': tx_bytes / (1024**2) if tx_bytes is not None else None,
                    'pcie_rx_mb': rx_bytes / (1024**2) if rx_bytes is not None else None,
                    'process_count': process_count,
                    'process_mem_mb': process_mem_mb
                }
                
            except Exception as e:
                # If we can't get data for a GPU, record the error
                gpu_data = {
                    'gpu_id': i,
                    'error': str(e)
                }
            
            sample['gpus'].append(gpu_data)
        
        return sample
    
    def monitor(self, duration=None):
        """Main monitoring loop"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        start_time = time.time()
        
        print(f"Starting GPU monitoring on {self.device_count} GPUs...")
        print(f"Output file: {self.output_file}")
        print(f"Sample interval: {self.sample_interval * 1000:.0f}ms")
        if duration:
            print(f"Duration: {duration}s")
        print("Press Ctrl+C to stop...")
        
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
        nvml.nvmlShutdown()
        print(f"\nMonitoring complete. Results saved to {self.output_file}")
        
    def save_results(self):
        """Save results to JSON file"""
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_samples'] = len(self.results['samples'])
        self.results['duration_seconds'] = self.results['samples'][-1]['timestamp'] if self.results['samples'] else 0
        
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor GPU resources for contention analysis')
    parser.add_argument('output_file', help='Output JSON file for monitoring data')
    parser.add_argument('--duration', type=float, help='Monitoring duration in seconds (default: unlimited)')
    parser.add_argument('--interval', type=float, default=0.1, help='Sample interval in seconds (default: 0.1)')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(args.output_file, args.interval)
    monitor.monitor(args.duration)

if __name__ == '__main__':
    main()