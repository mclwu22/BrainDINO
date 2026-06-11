#!/usr/bin/env python3
"""
Re-plot Kaplan-Meier curves from existing results with custom titles.

This script reads risk_stratification_results.csv from each results folder
and generates a new Kaplan-Meier plot with a custom title.

=== HOW TO USE ===

1. Modify the TITLE_MAPPING dictionary below to set your desired backbone names
2. Run: python replot_kaplan_meier.py

The script will:
- Scan all folders in ./results/
- For each folder with risk_stratification_results.csv, generate a new plot
- Save as kaplan_meier_custom.png in the same folder

=== CUSTOMIZATION ===

Edit TITLE_MAPPING to change backbone display names:
    "meddinov3" -> "Med-DINOv2"  (or whatever you want)
    "dinov3" -> "DINOv2"
    etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts


# ============================================================================
# MODIFY THIS SECTION TO CHANGE BACKBONE NAMES IN TITLES
# ============================================================================

TITLE_MAPPING = {
    # Format: "folder_prefix": "Display Name"
    # The script matches folder names that START WITH these keys

    "meddinov3": "BrainDINO",      # Change this to your desired name
    "dinov3": "DINOv3",              # Change this to your desired name
    "BrainMVP": "BrainMVP",          # Change this to your desired name
    "bm_mae": "BM-MAE",              # Change this to your desired name
}
# ============================================================================
# Kaplan–Meier visualization settings (paper-ready)
# ============================================================================

MAX_TIME_DAYS = 2000          # x-axis truncation for visualization
AT_RISK_TIMES = [0, 500, 1000, 1500, 2000]

# Output filename (saved in each results folder)
OUTPUT_FILENAME = "kaplan_meier_custom.png"

# ============================================================================
# END OF CUSTOMIZATION SECTION
# ============================================================================


def get_display_name(folder_name):
    """Get display name for a folder based on TITLE_MAPPING."""
    for prefix, display_name in TITLE_MAPPING.items():
        if folder_name.startswith(prefix):
            return display_name
    # Fallback: use folder name as-is
    return folder_name


def get_data_ratio_text(folder_name):
    """Extract data ratio from folder name if present."""
    if "_ratio" in folder_name:
        # Extract ratio number (e.g., "ratio20" -> "20%")
        ratio_part = folder_name.split("_ratio")[-1]
        try:
            ratio_num = int(ratio_part)
            return f" ({ratio_num}% Training Data)"
        except ValueError:
            pass
    return ""


def compute_logrank_pvalue(survival_times, events, risk_groups):
    """Compute log-rank test p-value for comparing survival curves."""
    mask_low = (risk_groups == 0)
    mask_high = (risk_groups == 1)

    if mask_low.sum() == 0 or mask_high.sum() == 0:
        return np.nan

    results = logrank_test(
        durations_A=survival_times[mask_low],
        durations_B=survival_times[mask_high],
        event_observed_A=events[mask_low],
        event_observed_B=events[mask_high]
    )
    return results.p_value


def plot_kaplan_meier(survival_times, events, risk_groups, save_path, title):
    """Plot Kaplan-Meier survival curves for high vs low risk groups
       with visualization-only x-axis truncation.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    # Masks
    mask_low = (risk_groups == 0)
    mask_high = (risk_groups == 1)

    # Fit KM (FULL follow-up, no truncation)
    if mask_low.sum() > 0:
        kmf_low.fit(
            survival_times[mask_low],
            events[mask_low],
            label=f'Low Risk (n={mask_low.sum()})'
        )
        kmf_low.plot_survival_function(
            ax=ax, ci_show=True, color='blue', linewidth=2
        )

    if mask_high.sum() > 0:
        kmf_high.fit(
            survival_times[mask_high],
            events[mask_high],
            label=f'High Risk (n={mask_high.sum()})'
        )
        kmf_high.plot_survival_function(
            ax=ax, ci_show=True, color='red', linewidth=2
        )

    # ---- Visualization truncation ONLY ----
    ax.set_xlim(0, MAX_TIME_DAYS)

    # # Add at-risk table (restricted time points)
    # fitted_kmfs = []
    # if mask_low.sum() > 0:
    #     fitted_kmfs.append(kmf_low)
    # if mask_high.sum() > 0:
    #     fitted_kmfs.append(kmf_high)

    # if fitted_kmfs:
    #     add_at_risk_counts(
    #         *fitted_kmfs,
    #         ax=ax,
    #         xticks=AT_RISK_TIMES
    #     )

    # Log-rank test (FULL data)
    p_value = compute_logrank_pvalue(survival_times, events, risk_groups)

    # Median survival (from full KM)
    median_low = kmf_low.median_survival_time_ if mask_low.sum() > 0 else None
    median_high = kmf_high.median_survival_time_ if mask_high.sum() > 0 else None

    # Labels & title - only show p-value
    ax.set_xlabel('Time from Surgery (days)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)

    # Format p-value
    if p_value < 0.001:
        p_text = "log-rank test, p < 0.001"
    else:
        p_text = f"log-rank test, p={p_value:.3f}"

    ax.set_title(p_text, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # # Footnote (审稿人防御句)
    # ax.text(
    #     0.01, -0.28,
    #     'X-axis truncated at 2000 days for visualization; '
    #     'log-rank statistics computed using full follow-up.',
    #     transform=ax.transAxes,
    #     fontsize=9,
    #     ha='left'
    # )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return p_value


def main():
    # Get the results directory (same level as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("=" * 70)
    print("Kaplan-Meier Re-plotting Script")
    print("=" * 70)
    print(f"\nResults directory: {results_dir}")
    print(f"Output filename: {OUTPUT_FILENAME}")
    print("\nTitle mapping:")
    for prefix, name in TITLE_MAPPING.items():
        print(f"  {prefix} -> {name}")
    print()

    # Find all subfolders with risk_stratification_results.csv
    processed = 0
    skipped = 0

    for folder_name in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        csv_path = os.path.join(folder_path, "risk_stratification_results.csv")

        if not os.path.exists(csv_path):
            print(f"[SKIP] {folder_name}: No risk_stratification_results.csv")
            skipped += 1
            continue

        # Load CSV
        df = pd.read_csv(csv_path)

        # Check required columns
        required_cols = ['survival_time', 'event', 'risk_group']
        if not all(col in df.columns for col in required_cols):
            print(f"[SKIP] {folder_name}: Missing required columns")
            skipped += 1
            continue

        # Build title
        display_name = get_display_name(folder_name)
        ratio_text = get_data_ratio_text(folder_name)
        title = f"Kaplan-Meier Curve - {display_name}"

        # Extract data
        survival_times = df['survival_time'].values
        events = df['event'].values
        risk_groups = df['risk_group'].values

        # Plot
        output_path = os.path.join(folder_path, OUTPUT_FILENAME)
        p_value = plot_kaplan_meier(
            survival_times=survival_times,
            events=events,
            risk_groups=risk_groups,
            save_path=output_path,
            title=title
        )

        print(f"[OK] {folder_name}")
        print(f"     Title: {title}")
        print(f"     p-value: {p_value:.4f}")
        print(f"     Saved: {output_path}")
        processed += 1

    print()
    print("=" * 70)
    print(f"Done! Processed: {processed}, Skipped: {skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
