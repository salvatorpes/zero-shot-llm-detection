import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Create result folder if it doesn't exist
Path("result").mkdir(exist_ok=True)

# Professional color palette (colorblind-friendly)
CB_PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999", "#E69F00", "#000000"]

# Set global plot style
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5,
})

# Baseline/default performance values
BASELINE_ROC_AUC = 0.9941
BASELINE_PR_AUC = 0.9948

def load_data(path):
    """Load and organize data from CSV file"""
    data_by_limit = defaultdict(list)
    data_by_position = defaultdict(list)
    
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pos = int(row["position"])
            lim = int(row["limit"])
            roc = float(row["roc_auc"])
            pr = float(row["pr_auc"])
            data_by_limit[lim].append((pos, roc, pr))
            data_by_position[pos].append((lim, roc, pr))
    
    # Sort data
    for k in data_by_limit:
        data_by_limit[k].sort(key=lambda x: x[0])
    for k in data_by_position:
        data_by_position[k].sort(key=lambda x: x[0])
    
    return data_by_limit, data_by_position

def plot_roc_vs_pos(data_by_limit, out="result/roc_vs_pos_by_k.pdf"):
    """Plot ROC-AUC vs Number of Positives (lines by Cutoff-K)"""
    fig, ax = plt.subplots()
    
    for i, k in enumerate(sorted(data_by_limit)):
        xs = [p for p, _, __ in data_by_limit[k]]
        ys = [r for _, r, __ in data_by_limit[k]]
        ax.plot(xs, ys, marker="o", markersize=4, 
                label=f"Cutoff-K = {k}", 
                color=CB_PALETTE[i % len(CB_PALETTE)])
    
    # Add baseline
    ax.axhline(BASELINE_ROC_AUC, ls="--", lw=2, color="#4D4D4D", 
               label=f"Baseline ROC AUC = {BASELINE_ROC_AUC}")
    ax.text(0.99, BASELINE_ROC_AUC + 0.005, "Baseline", 
            color="#4D4D4D", ha="right", va="bottom", 
            transform=ax.get_yaxis_transform())
    
    ax.set(
        title="ROC-AUC vs Number of Positives",
        xlabel="Num of Pos",
        ylabel="ROC-AUC",
        ylim=(0, 1.0)
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)

def plot_pr_vs_pos(data_by_limit, out="result/pr_vs_pos_by_k.pdf"):
    """Plot PR-AUC vs Number of Positives (lines by Cutoff-K)"""
    fig, ax = plt.subplots()
    
    for i, k in enumerate(sorted(data_by_limit)):
        xs = [p for p, _, __ in data_by_limit[k]]
        ys = [pr for _, __, pr in data_by_limit[k]]
        ax.plot(xs, ys, marker="s", markersize=4, 
                label=f"Cutoff-K = {k}", 
                color=CB_PALETTE[i % len(CB_PALETTE)])
    
    # Add baseline
    ax.axhline(BASELINE_PR_AUC, ls="--", lw=2, color="#4D4D4D", 
               label=f"Baseline PR AUC = {BASELINE_PR_AUC}")
    ax.text(0.99, BASELINE_PR_AUC + 0.005, "Baseline", 
            color="#4D4D4D", ha="right", va="bottom", 
            transform=ax.get_yaxis_transform())
    
    ax.set(
        title="PR-AUC vs Number of Positives",
        xlabel="Num of Pos",
        ylabel="PR-AUC",
        ylim=(0, 1.0)
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)

def plot_roc_vs_k(data_by_position, out="result/roc_vs_k_by_pos.pdf"):
    """Plot ROC-AUC vs Cutoff-K (lines by selected positions)"""
    fig, ax = plt.subplots()
    
    # Select subset of positions to avoid clutter
    all_positions = sorted(data_by_position.keys())
    selected_positions = all_positions[::max(1, len(all_positions) // 8)]
    
    for i, pos in enumerate(selected_positions):
        xs = [k for k, _, __ in data_by_position[pos]]
        ys = [r for _, r, __ in data_by_position[pos]]
        ax.plot(xs, ys, marker="o", markersize=4, 
                label=f"Num of Pos = {pos}", 
                color=CB_PALETTE[i % len(CB_PALETTE)])
    
    # Add baseline
    ax.axhline(BASELINE_ROC_AUC, ls="--", lw=2, color="#4D4D4D", 
               label=f"Baseline ROC AUC = {BASELINE_ROC_AUC}")
    ax.text(0.99, BASELINE_ROC_AUC + 0.005, "Baseline", 
            color="#4D4D4D", ha="right", va="bottom", 
            transform=ax.get_yaxis_transform())
    
    ax.set(
        title="ROC-AUC vs Cutoff-K",
        xlabel="Cutoff-K",
        ylabel="ROC-AUC",
        ylim=(0, 1.0)
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)

def plot_pr_vs_k(data_by_position, out="result/pr_vs_k_by_pos.pdf"):
    """Plot PR-AUC vs Cutoff-K (lines by selected positions)"""
    fig, ax = plt.subplots()
    
    # Select subset of positions to avoid clutter
    all_positions = sorted(data_by_position.keys())
    selected_positions = all_positions[::max(1, len(all_positions) // 8)]
    
    for i, pos in enumerate(selected_positions):
        xs = [k for k, _, __ in data_by_position[pos]]
        ys = [pr for _, __, pr in data_by_position[pos]]
        ax.plot(xs, ys, marker="s", markersize=4, 
                label=f"Num of Pos = {pos}", 
                color=CB_PALETTE[i % len(CB_PALETTE)])
    
    # Add baseline
    ax.axhline(BASELINE_PR_AUC, ls="--", lw=2, color="#4D4D4D", 
               label=f"Baseline PR AUC = {BASELINE_PR_AUC}")
    ax.text(0.99, BASELINE_PR_AUC + 0.005, "Baseline", 
            color="#4D4D4D", ha="right", va="bottom", 
            transform=ax.get_yaxis_transform())
    
    ax.set(
        title="PR-AUC vs Cutoff-K",
        xlabel="Cutoff-K",
        ylabel="PR-AUC",
        ylim=(0, 1.0)
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)

def create_combined_plot(data_by_limit, data_by_position, out="result/combined_plots.pdf"):
    """Create 2x2 subplot with all four plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: ROC vs Pos
    for i, k in enumerate(sorted(data_by_limit)):
        xs = [p for p, _, __ in data_by_limit[k]]
        ys = [r for _, r, __ in data_by_limit[k]]
        ax1.plot(xs, ys, marker="o", markersize=4, 
                label=f"Cutoff-K = {k}", 
                color=CB_PALETTE[i % len(CB_PALETTE)])
    ax1.axhline(BASELINE_ROC_AUC, ls="--", lw=2, color="#4D4D4D", 
                label=f"Baseline ROC AUC = {BASELINE_ROC_AUC}")
    ax1.set(title="ROC-AUC vs Number of Positives", 
            xlabel="Num of Pos", ylabel="ROC-AUC", ylim=(0, 1.0))
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)
    
    # Plot 2: PR vs Pos
    for i, k in enumerate(sorted(data_by_limit)):
        xs = [p for p, _, __ in data_by_limit[k]]
        ys = [pr for _, __, pr in data_by_limit[k]]
        ax2.plot(xs, ys, marker="s", markersize=4, 
                label=f"Cutoff-K = {k}", 
                color=CB_PALETTE[i % len(CB_PALETTE)])
    ax2.axhline(BASELINE_PR_AUC, ls="--", lw=2, color="#4D4D4D", 
                label=f"Baseline PR AUC = {BASELINE_PR_AUC}")
    ax2.set(title="PR-AUC vs Number of Positives", 
            xlabel="Num of Pos", ylabel="PR-AUC", ylim=(0, 1.0))
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="best", fontsize=9, framealpha=0.9)
    
    # Plot 3: ROC vs K
    all_positions = sorted(data_by_position.keys())
    selected_positions = all_positions[::max(1, len(all_positions) // 8)]
    for i, pos in enumerate(selected_positions):
        xs = [k for k, _, __ in data_by_position[pos]]
        ys = [r for _, r, __ in data_by_position[pos]]
        ax3.plot(xs, ys, marker="o", markersize=4, 
                label=f"Num of Pos = {pos}", 
                color=CB_PALETTE[i % len(CB_PALETTE)])
    ax3.axhline(BASELINE_ROC_AUC, ls="--", lw=2, color="#4D4D4D", 
                label=f"Baseline ROC AUC = {BASELINE_ROC_AUC}")
    ax3.set(title="ROC-AUC vs Cutoff-K", 
            xlabel="Cutoff-K", ylabel="ROC-AUC", ylim=(0, 1.0))
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="best", fontsize=9, framealpha=0.9)
    
    # Plot 4: PR vs K
    for i, pos in enumerate(selected_positions):
        xs = [k for k, _, __ in data_by_position[pos]]
        ys = [pr for _, __, pr in data_by_position[pos]]
        ax4.plot(xs, ys, marker="s", markersize=4, 
                label=f"Num of Pos = {pos}", 
                color=CB_PALETTE[i % len(CB_PALETTE)])
    ax4.axhline(BASELINE_PR_AUC, ls="--", lw=2, color="#4D4D4D", 
                label=f"Baseline PR AUC = {BASELINE_PR_AUC}")
    ax4.set(title="PR-AUC vs Cutoff-K", 
            xlabel="Cutoff-K", ylabel="PR-AUC", ylim=(0, 1.0))
    ax4.grid(True, alpha=0.2)
    ax4.legend(loc="best", fontsize=9, framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data_by_limit, data_by_position = load_data('experiment_results_combined.csv')
    
    print(f"Found {len(data_by_limit)} unique Cutoff-K values")
    print(f"Found {len(data_by_position)} unique position values")
    
    # Create all plots
    print("\nGenerating plots...")
    plot_roc_vs_pos(data_by_limit)
    plot_pr_vs_pos(data_by_limit)
    plot_roc_vs_k(data_by_position)
    plot_pr_vs_k(data_by_position)
    create_combined_plot(data_by_limit, data_by_position)
    
    print("\n✓ All plots generated successfully!")
    print("\nOutput files:")
    print("  - roc_vs_pos_by_k.pdf")
    print("  - pr_vs_pos_by_k.pdf")
    print("  - roc_vs_k_by_pos.pdf")
    print("  - pr_vs_k_by_pos.pdf")
    print("  - combined_plots.pdf")
