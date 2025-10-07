import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
    
    print("\n=== Creating Individual Line Plots ===")
    create_individual_line_plots()