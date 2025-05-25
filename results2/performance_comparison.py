#!/usr/bin/env python3
"""
Performance Comparison Analysis: DRL Agent Dimensional Fix
Compares results before and after fixing the state size mismatch
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_metrics(filepath):
    """Load metrics from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_comparison_analysis():
    """Create comprehensive comparison analysis"""
    
    # Load results from different runs
    results = {
        'Before Fix (090321)': load_metrics('final_metrics_20250525_090321.json'),
        'After Fix (091639)': load_metrics('results/final_metrics_20250525_091639.json')
    }
    
    # Remove None results
    results = {k: v for k, v in results.items() if v is not None}
    
    if len(results) < 2:
        print("Insufficient data for comparison")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=200)
    fig.suptitle('DRL Agent Performance: Before vs After Dimensional Fix', fontsize=20, fontweight='bold')
    
    # Extract data for plotting
    run_names = list(results.keys())
    metrics = ['alive_percentage', 'avg_energy', 'total_traffic', 'sink_connectivity', 'energy_variance', 'num_cluster_heads']
    metric_labels = ['Alive Nodes (%)', 'Avg Energy (J)', 'Total Traffic', 'Sink Connectivity (%)', 'Energy Variance', 'Cluster Heads']
    
    colors = ['#FF6B6B', '#4ECDC4']
    hatch_patterns = ['/', 'O']
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = [results[run][metric] for run in run_names]
        bars = ax.bar(run_names, values, color=colors[:len(run_names)], alpha=0.85, edgecolor='black', linewidth=2, hatch=hatch_patterns[idx%2])
        
        ax.set_title(label, fontweight='bold', fontsize=16)
        ax.set_ylabel(label, fontsize=14)
        ax.grid(True, alpha=0.25, linestyle='--')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}' if isinstance(value, float) else f'{value}',
                   ha='center', va='bottom', fontweight='bold', fontsize=13)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=15, labelsize=13)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('results/performance_comparison_dimensional_fix.png', dpi=400, bbox_inches='tight')
    plt.savefig('results/performance_comparison_dimensional_fix.svg', bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison
    print("=" * 80)
    print("DETAILED PERFORMANCE COMPARISON")
    print("=" * 80)
    
    for run_name, data in results.items():
        print(f"\n{run_name}:")
        print(f"  Alive Nodes: {data['alive_nodes']}/100 ({data['alive_percentage']:.1f}%)")
        print(f"  Average Energy: {data['avg_energy']:.3f}J")
        print(f"  Total Traffic: {data['total_traffic']}")
        print(f"  Sink Connectivity: {data['sink_connectivity']:.1f}%")
        print(f"  Network Connectivity: {data['connectivity']}")
        print(f"  Energy Variance: {data['energy_variance']:.3f}")
        print(f"  Cluster Heads: {data['num_cluster_heads']}")
    
    # Calculate improvements
    if len(results) == 2:
        before = list(results.values())[0]
        after = list(results.values())[1]
        
        print(f"\n{'='*80}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'='*80}")
        
        metrics_to_compare = [
            ('alive_percentage', 'Node Survival Rate', '%'),
            ('avg_energy', 'Energy Efficiency', 'J'),
            ('total_traffic', 'Traffic Throughput', 'packets'),
            ('sink_connectivity', 'Sink Connectivity', '%'),
            ('energy_variance', 'Energy Distribution', 'variance')
        ]
        
        for metric, name, unit in metrics_to_compare:
            before_val = before[metric]
            after_val = after[metric]
            
            if before_val != 0:
                change_pct = ((after_val - before_val) / before_val) * 100
                direction = "↗" if change_pct > 0 else "↘" if change_pct < 0 else "→"
                print(f"{name:.<25} {before_val:>8.2f} → {after_val:>8.2f} {unit:<10} ({direction} {abs(change_pct):>6.1f}%)")
            else:
                print(f"{name:.<25} {before_val:>8.2f} → {after_val:>8.2f} {unit}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print("The dimensional fix (state_size: 7 → 9) resolved the tensor mismatch error")
    print("and allowed the DRL agent to properly process all 9 state features:")
    print("1. Energy level")
    print("2. X coordinate") 
    print("3. Y coordinate")
    print("4. Distance to sink")
    print("5. Hop count")
    print("6. Network energy")
    print("7. Congestion level")
    print("8. Sleep state")
    print("9. Duty cycle")
    print(f"\nResults generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    create_comparison_analysis()
