import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def create_line_plots():
    """Create line plots showing trends with separate lines for each parameter value"""
    
    # Baseline/default performance values
    BASELINE_ROC_AUC = 0.9941
    BASELINE_PR_AUC = 0.9948
    
    # Read the CSV file manually
    data = []
    with open('experiment_results_combined.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'position': int(row['position']),
                'limit': int(row['limit']),
                'roc_auc': float(row['roc_auc']),
                'pr_auc': float(row['pr_auc'])
            })
    
    if len(data) == 0:
        print("No data found!")
        return
    
    # Organize data by position and limit
    data_by_limit = defaultdict(list)  # limit -> list of (position, roc_auc, pr_auc)
    data_by_position = defaultdict(list)  # position -> list of (limit, roc_auc, pr_auc)
    
    for row in data:
        data_by_limit[row['limit']].append((row['position'], row['roc_auc'], row['pr_auc']))
        data_by_position[row['position']].append((row['limit'], row['roc_auc'], row['pr_auc']))
    
    # Sort data within each group
    for limit in data_by_limit:
        data_by_limit[limit].sort(key=lambda x: x[0])  # sort by position
    
    for position in data_by_position:
        data_by_position[position].sort(key=lambda x: x[0])  # sort by limit
    
    # Get unique values
    unique_limits = sorted(data_by_limit.keys())
    unique_positions = sorted(data_by_position.keys())
    
    # Define colors for lines - simple and reliable
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Create figure with 4 subplots (2x2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # =================================================================
    # Plot 1: Position vs ROC AUC (separate line for each limit)
    # =================================================================
    ax1.set_title('ROC AUC vs Num of Pos (Lines by Cutoff-K)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Num of Pos', fontsize=12)
    ax1.set_ylabel('ROC AUC', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for i, limit in enumerate(unique_limits):
        positions = [item[0] for item in data_by_limit[limit]]
        roc_values = [item[1] for item in data_by_limit[limit]]
        
        ax1.plot(positions, roc_values, marker='o', linewidth=2, markersize=6, 
                color=colors[i], label=f'Cutoff-K {limit}')
    
    # Add baseline line
    all_positions = sorted(set(item[0] for sublist in data_by_limit.values() for item in sublist))
    ax1.axhline(y=BASELINE_ROC_AUC, color='red', linestyle='--', linewidth=3, 
                label=f'Baseline (ROC: {BASELINE_ROC_AUC})', alpha=0.8)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_ylim(0, 1)
    
    # =================================================================
    # Plot 2: Position vs PR AUC (separate line for each limit)
    # =================================================================
    ax2.set_title('PR AUC vs Num of Pos (Lines by Cutoff-K)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Num of Pos', fontsize=12)
    ax2.set_ylabel('PR AUC', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for i, limit in enumerate(unique_limits):
        positions = [item[0] for item in data_by_limit[limit]]
        pr_values = [item[2] for item in data_by_limit[limit]]
        
        ax2.plot(positions, pr_values, marker='s', linewidth=2, markersize=6, 
                color=colors[i], label=f'Cutoff-K {limit}')
    
    # Add baseline line
    ax2.axhline(y=BASELINE_PR_AUC, color='red', linestyle='--', linewidth=3, 
                label=f'Baseline (PR: {BASELINE_PR_AUC})', alpha=0.8)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 1)
    
    # =================================================================
    # Plot 3: Limit vs ROC AUC (separate line for each position)
    # =================================================================
    ax3.set_title('ROC AUC vs Cutoff-K (Lines by Num of Pos)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Cutoff-K', fontsize=12)
    ax3.set_ylabel('ROC AUC', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Use fewer position lines to avoid clutter
    selected_positions = unique_positions[::max(1, len(unique_positions)//8)]  # Show max 8 lines
    
    for i, position in enumerate(selected_positions):
        limits = [item[0] for item in data_by_position[position]]
        roc_values = [item[1] for item in data_by_position[position]]
        
        ax3.plot(limits, roc_values, marker='o', linewidth=2, markersize=6, 
                color=colors[i % len(colors)], label=f'Num of Pos {position}')
    
    # Add baseline line
    ax3.axhline(y=BASELINE_ROC_AUC, color='red', linestyle='--', linewidth=3, 
                label=f'Baseline (ROC: {BASELINE_ROC_AUC})', alpha=0.8)
    
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_ylim(0, 1)
    
    # =================================================================
    # Plot 4: Limit vs PR AUC (separate line for each position)
    # =================================================================
    ax4.set_title('PR AUC vs Cutoff-K (Lines by Num of Pos)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Cutoff-K', fontsize=12)
    ax4.set_ylabel('PR AUC', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    for i, position in enumerate(selected_positions):
        limits = [item[0] for item in data_by_position[position]]
        pr_values = [item[2] for item in data_by_position[position]]
        
        ax4.plot(limits, pr_values, marker='s', linewidth=2, markersize=6, 
                color=colors[i % len(colors)], label=f'Num of Pos {position}')
    
    # Add baseline line
    ax4.axhline(y=BASELINE_PR_AUC, color='red', linestyle='--', linewidth=3, 
                label=f'Baseline (PR: {BASELINE_PR_AUC})', alpha=0.8)
    
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_ylim(0, 1)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('line_plots_by_parameters.pdf', dpi=300, bbox_inches='tight')
    print("Line plots saved as 'line_plots_by_parameters.pdf'")
    
    # Show the plot
    plt.show()
    
    # Print summary information
    print(f"\nData Summary:")
    print(f"Total data points: {len(data)}")
    print(f"Unique positions: {unique_positions}")
    print(f"Unique limits: {unique_limits}")
    print(f"Position range: {min(unique_positions)} to {max(unique_positions)}")
    print(f"Limit range: {min(unique_limits)} to {max(unique_limits)}")


def create_perplexity_line_plots(results_subdir=None, top_left_only_path=None):
    """Create line plots for mean sampled perplexity across position and limit combinations.

    Args:
        results_subdir: Optional path to the directory containing sampling discrepancy JSON files.
        top_left_only_path: If provided, only the top-left subplot (mean sampled perplexity vs
            number of positions) is generated and saved to the specified PDF path.
    """

    script_dir = Path(__file__).resolve().parent
    if results_subdir is None:
        results_path = (script_dir.parent / 'exp_main' / 'results_perplexity').resolve()
    else:
        results_path = Path(results_subdir)
        if not results_path.is_absolute():
            results_path = (script_dir / results_path).resolve()
        if not results_path.exists():
            alt_path = (script_dir.parent / results_subdir).resolve()
            if alt_path.exists():
                results_path = alt_path
    if not results_path.exists():
        legacy_path = (script_dir.parent / 'exp_main' / 'results_test').resolve()
        if legacy_path.exists():
            results_path = legacy_path

    if not results_path.exists():
        print(f"Perplexity results directory not found: {results_path}")
        return

    pattern = re.compile(r'pos(\d+)_limit(\d+)')
    data = []

    for json_path in results_path.glob('*.sampling_discrepancy.json'):
        match = pattern.search(json_path.name)
        if not match:
            continue

        position = int(match.group(1))
        limit = int(match.group(2))

        with json_path.open('r', encoding='utf-8') as f:
            payload = json.load(f)

        metrics = payload.get('metrics') or {}

        mean_sample = metrics.get('mean_sampled_perplexity') or payload.get('mean_sampled_perplexity')
        if mean_sample is None:
            samples = payload.get('samples_perplexity')
            if samples:
                mean_sample = float(np.mean(samples))

        mean_original = metrics.get('mean_original_perplexity') or payload.get('mean_original_perplexity')
        if mean_original is None:
            originals = payload.get('original_perplexity') or payload.get('real_perplexity')
            if originals:
                mean_original = float(np.mean(originals))

        if mean_sample is None:
            continue

        data.append({
            'position': position,
            'limit': limit,
            'mean_sample': float(mean_sample),
            'mean_original': float(mean_original) if mean_original is not None else None
        })

    if not data:
        print("No perplexity data found!")
        return

    data_by_limit = defaultdict(list)
    data_by_position = defaultdict(list)

    for row in data:
        data_by_limit[row['limit']].append((row['position'], row['mean_sample'], row['mean_original']))
        data_by_position[row['position']].append((row['limit'], row['mean_sample'], row['mean_original']))

    for limit in data_by_limit:
        data_by_limit[limit].sort(key=lambda x: x[0])

    for position in data_by_position:
        data_by_position[position].sort(key=lambda x: x[0])

    unique_limits = sorted(data_by_limit.keys())
    unique_positions = sorted(data_by_position.keys())
    selected_positions = unique_positions[::max(1, len(unique_positions)//8)] or unique_positions

    all_sample_values = [row['mean_sample'] for row in data]
    all_original_values = [row['mean_original'] for row in data if row['mean_original'] is not None]
    min_original = max_original = None
    if all_original_values:
        min_original = min(all_original_values)
        max_original = max(all_original_values)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    min_sample = min(all_sample_values)
    max_sample = max(all_sample_values)
    sample_margin = (max_sample - min_sample) * 0.1 if max_sample > min_sample else 1.0

    def plot_sample_vs_position(ax):
        ax.set_title('Mean Sampled Perplexity vs Num of Pos (Lines by Cutoff-K)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Num of Pos', fontsize=12)
        ax.set_ylabel('Mean Sampled Perplexity', fontsize=12)
        ax.grid(True, alpha=0.3)
        for i, limit in enumerate(unique_limits):
            positions = [item[0] for item in data_by_limit[limit]]
            sample_values = [item[1] for item in data_by_limit[limit]]
            ax.plot(positions, sample_values, marker='o', linewidth=2, markersize=6,
                    color=colors[i % len(colors)], label=f'Cutoff-K {limit}')
        ax.set_ylim(min_sample - sample_margin, max_sample + sample_margin)
        ax.legend(loc='upper left', frameon=True)

    if top_left_only_path:
        fig, ax = plt.subplots(figsize=(10, 7))
        plot_sample_vs_position(ax)
        plt.tight_layout()
        output_path = Path(top_left_only_path)
        if not output_path.is_absolute():
            output_path = (script_dir / output_path).resolve()
        else:
            output_path = output_path.resolve()
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"Perplexity line plot saved as '{output_path.name}'")
        plt.show()
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        plot_sample_vs_position(ax1)

        ax2.set_title('Mean Sampled Perplexity vs Cutoff-K (Lines by Num of Pos)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cutoff-K', fontsize=12)
        ax2.set_ylabel('Mean Sampled Perplexity', fontsize=12)
        ax2.grid(True, alpha=0.3)

        for i, position in enumerate(selected_positions):
            limits = [item[0] for item in data_by_position[position]]
            sample_values = [item[1] for item in data_by_position[position]]
            ax2.plot(limits, sample_values, marker='s', linewidth=2, markersize=6,
                     color=colors[i % len(colors)], label=f'Num of Pos {position}')

        ax2.set_ylim(min_sample - sample_margin, max_sample + sample_margin)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax3.set_title('Mean Original Perplexity vs Num of Pos (Lines by Cutoff-K)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Num of Pos', fontsize=12)
        ax3.set_ylabel('Mean Original Perplexity', fontsize=12)
        ax3.grid(True, alpha=0.3)

        if min_original is not None and max_original is not None:
            original_margin = (max_original - min_original) * 0.1 if max_original > min_original else 1.0
            for i, limit in enumerate(unique_limits):
                positions = [item[0] for item in data_by_limit[limit] if item[2] is not None]
                original_values = [item[2] for item in data_by_limit[limit] if item[2] is not None]
                if positions:
                    ax3.plot(positions, original_values, marker='o', linewidth=2, markersize=6,
                             color=colors[i % len(colors)], label=f'Cutoff-K {limit}')
            ax3.set_ylim(min_original - original_margin, max_original + original_margin)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax3.text(0.5, 0.5, 'No original perplexity data available', transform=ax3.transAxes,
                     ha='center', va='center', fontsize=12)

        ax4.set_title('Original - Sampled Perplexity Δ vs Cutoff-K (Lines by Num of Pos)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Cutoff-K', fontsize=12)
        ax4.set_ylabel('Perplexity Δ (Original - Sampled)', fontsize=12)
        ax4.grid(True, alpha=0.3)

        deltas = []
        for i, position in enumerate(selected_positions):
            limits = []
            delta_values = []
            for item in data_by_position[position]:
                if item[2] is None:
                    continue
                limits.append(item[0])
                delta_values.append(item[2] - item[1])
            if limits:
                deltas.extend(delta_values)
                ax4.plot(limits, delta_values, marker='^', linewidth=2, markersize=6,
                         color=colors[i % len(colors)], label=f'Num of Pos {position}')

        if deltas:
            delta_margin = (max(deltas) - min(deltas)) * 0.1 if max(deltas) > min(deltas) else 0.5
            ax4.set_ylim(min(deltas) - delta_margin, max(deltas) + delta_margin)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for delta computation', transform=ax4.transAxes,
                     ha='center', va='center', fontsize=12)

        plt.tight_layout()
        output_path = script_dir / 'perplexity_line_plots.pdf'
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"Perplexity line plots saved as '{output_path.name}'")
        plt.show()

    print("\nPerplexity Data Summary:")
    print(f"Total combinations: {len(data)}")
    print(f"Unique positions: {unique_positions}")
    print(f"Unique limits: {unique_limits}")
    print(f"Mean sampled perplexity range: {min_sample:.3f} to {max_sample:.3f}")
    if all_original_values:
        print(f"Mean original perplexity range: {min_original:.3f} to {max_original:.3f}")

def create_individual_line_plots():
    """Create separate plots for each metric and parameter combination"""
    
    # Baseline/default performance values
    BASELINE_ROC_AUC = 0.9941
    BASELINE_PR_AUC = 0.9948
    
    # Read data
    data = []
    with open('experiment_results_combined.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'position': int(row['position']),
                'limit': int(row['limit']),
                'roc_auc': float(row['roc_auc']),
                'pr_auc': float(row['pr_auc'])
            })
    
    # Organize data
    data_by_limit = defaultdict(list)
    data_by_position = defaultdict(list)
    
    for row in data:
        data_by_limit[row['limit']].append((row['position'], row['roc_auc'], row['pr_auc']))
        data_by_position[row['position']].append((row['limit'], row['roc_auc'], row['pr_auc']))
    
    # Sort data
    for limit in data_by_limit:
        data_by_limit[limit].sort(key=lambda x: x[0])
    for position in data_by_position:
        data_by_position[position].sort(key=lambda x: x[0])
    
    unique_limits = sorted(data_by_limit.keys())
    unique_positions = sorted(data_by_position.keys())
    
    # Define colors for lines - simple and reliable (same as combined plots)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # ===============================================================
    # Individual Plot 1: ROC AUC vs Position (by Limit)
    # ===============================================================
    plt.figure(figsize=(12, 8))
    for i, limit in enumerate(unique_limits):
        positions = [item[0] for item in data_by_limit[limit]]
        roc_values = [item[1] for item in data_by_limit[limit]]
        plt.plot(positions, roc_values, marker='o', linewidth=3, markersize=8, 
                color=colors[i], label=f'Cutoff-K {limit}')
    
    # Add baseline line
    plt.axhline(y=BASELINE_ROC_AUC, color='red', linestyle='--', linewidth=3, 
                label=f'Baseline (ROC: {BASELINE_ROC_AUC})', alpha=0.8)
    
    plt.title('ROC AUC vs Num of Pos (Separate Lines by Cutoff-K)', fontsize=16, fontweight='bold')
    plt.xlabel('Num of Pos', fontsize=14)
    plt.ylabel('ROC AUC', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('roc_vs_position_by_limit.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===============================================================
    # Individual Plot 2: PR AUC vs Position (by Limit)
    # ===============================================================
    plt.figure(figsize=(12, 8))
    for i, limit in enumerate(unique_limits):
        positions = [item[0] for item in data_by_limit[limit]]
        pr_values = [item[2] for item in data_by_limit[limit]]
        plt.plot(positions, pr_values, marker='s', linewidth=3, markersize=8, 
                color=colors[i], label=f'Cutoff-K {limit}')
    
    # Add baseline line
    plt.axhline(y=BASELINE_PR_AUC, color='red', linestyle='--', linewidth=3, 
                label=f'Baseline (PR: {BASELINE_PR_AUC})', alpha=0.8)
    
    plt.title('PR AUC vs Num of Pos (Separate Lines by Cutoff-K)', fontsize=16, fontweight='bold')
    plt.xlabel('Num of Pos', fontsize=14)
    plt.ylabel('PR AUC', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('pr_vs_position_by_limit.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===============================================================
    # Individual Plot 3: ROC AUC vs Limit (by Position - selected)
    # ===============================================================
    selected_positions = unique_positions[::max(1, len(unique_positions)//8)]
    
    plt.figure(figsize=(12, 8))
    for i, position in enumerate(selected_positions):
        limits = [item[0] for item in data_by_position[position]]
        roc_values = [item[1] for item in data_by_position[position]]
        plt.plot(limits, roc_values, marker='o', linewidth=3, markersize=8, 
                color=colors[i % len(colors)], label=f'Num of Pos {position}')
    
    # Add baseline line
    plt.axhline(y=BASELINE_ROC_AUC, color='red', linestyle='--', linewidth=3, 
                label=f'Baseline (ROC: {BASELINE_ROC_AUC})', alpha=0.8)
    
    plt.title('ROC AUC vs Cutoff-K (Separate Lines by Num of Pos)', fontsize=16, fontweight='bold')
    plt.xlabel('Cutoff-K', fontsize=14)
    plt.ylabel('ROC AUC', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('roc_vs_limit_by_position.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===============================================================
    # Individual Plot 4: PR AUC vs Limit (by Position - selected)
    # ===============================================================
    plt.figure(figsize=(12, 8))
    for i, position in enumerate(selected_positions):
        limits = [item[0] for item in data_by_position[position]]
        pr_values = [item[2] for item in data_by_position[position]]
        plt.plot(limits, pr_values, marker='s', linewidth=3, markersize=8, 
                color=colors[i % len(colors)], label=f'Num of Pos {position}')
    
    # Add baseline line
    plt.axhline(y=BASELINE_PR_AUC, color='red', linestyle='--', linewidth=3, 
                label=f'Baseline (PR: {BASELINE_PR_AUC})', alpha=0.8)
    
    plt.title('PR AUC vs Cutoff-K (Separate Lines by Num of Pos)', fontsize=16, fontweight='bold')
    plt.xlabel('Cutoff-K', fontsize=14)
    plt.ylabel('PR AUC', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('pr_vs_limit_by_position.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Individual line plots saved:")
    print("- roc_vs_position_by_limit.pdf")
    print("- pr_vs_position_by_limit.pdf") 
    print("- roc_vs_limit_by_position.pdf")
    print("- pr_vs_limit_by_position.pdf")

if __name__ == "__main__":
    print("=== Creating Combined Line Plots ===")
    create_line_plots()
    
    print("\n=== Creating Perplexity Line Plots ===")
    create_perplexity_line_plots()

    print("\n=== Creating Individual Line Plots ===")
    create_individual_line_plots()