import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts


def compute_median_risk_threshold(train_risk_scores):
    """
    Compute median risk score from training set.

    Args:
        train_risk_scores: np.array of risk scores from training set

    Returns:
        median: float, median risk score
    """
    median = np.median(train_risk_scores)
    return median


def stratify_patients(risk_scores, threshold):
    """
    Stratify patients into high/low risk groups based on threshold.

    Args:
        risk_scores: np.array of risk scores
        threshold: float, risk score threshold (typically median from train set)

    Returns:
        groups: np.array of group assignments (0=low risk, 1=high risk)
    """
    groups = (risk_scores >= threshold).astype(int)
    return groups


def plot_kaplan_meier(survival_times, events, risk_groups, save_path, title="Kaplan-Meier Survival Curves"):
    """
    Plot Kaplan-Meier survival curves for high vs low risk groups.

    Args:
        survival_times: np.array of survival times (days)
        events: np.array of event indicators (1=deceased, 0=censored)
        risk_groups: np.array of risk group assignments (0=low risk, 1=high risk)
        save_path: str, path to save the plot
        title: str, plot title

    Returns:
        fig: matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Fit KM curves for each group
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    # Low risk group
    mask_low = (risk_groups == 0)
    if mask_low.sum() > 0:
        kmf_low.fit(
            survival_times[mask_low],
            events[mask_low],
            label=f'Low Risk (n={mask_low.sum()})'
        )
        kmf_low.plot_survival_function(ax=ax, ci_show=True, color='blue', linewidth=2)

    # High risk group
    mask_high = (risk_groups == 1)
    if mask_high.sum() > 0:
        kmf_high.fit(
            survival_times[mask_high],
            events[mask_high],
            label=f'High Risk (n={mask_high.sum()})'
        )
        kmf_high.plot_survival_function(ax=ax, ci_show=True, color='red', linewidth=2)

    fitted_kmfs = []
    if mask_low.sum() > 0:
        fitted_kmfs.append(kmf_low)
    if mask_high.sum() > 0:
        fitted_kmfs.append(kmf_high)
    if fitted_kmfs:
        add_at_risk_counts(*fitted_kmfs, ax=ax)

    # Compute log-rank p-value
    p_value = compute_logrank_pvalue(survival_times, events, risk_groups)

    # Add median survival times to legend
    median_low = kmf_low.median_survival_time_ if mask_low.sum() > 0 else None
    median_high = kmf_high.median_survival_time_ if mask_high.sum() > 0 else None

    # Formatting
    ax.set_xlabel('Time from Surgery (days)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'{title}\nLog-rank p-value: {p_value:.4f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Add text box with median survival times
    textstr = ''
    if median_low is not None:
        textstr += f'Median survival (Low Risk): {median_low:.1f} days\n'
    if median_high is not None:
        textstr += f'Median survival (High Risk): {median_high:.1f} days'

    if textstr:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.60, 0.25, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Survival Utils] Kaplan-Meier plot saved to {save_path}")

    return fig


def compute_logrank_pvalue(survival_times, events, risk_groups):
    """
    Compute log-rank test p-value for comparing survival curves.

    Args:
        survival_times: np.array of survival times (days)
        events: np.array of event indicators (1=deceased, 0=censored)
        risk_groups: np.array of risk group assignments (0=low risk, 1=high risk)

    Returns:
        p_value: float, log-rank test p-value
    """
    # Separate groups
    mask_low = (risk_groups == 0)
    mask_high = (risk_groups == 1)

    # Check if both groups have samples
    if mask_low.sum() == 0 or mask_high.sum() == 0:
        print("[WARNING] One of the groups is empty, cannot compute log-rank test")
        return np.nan

    # Perform log-rank test
    results = logrank_test(
        durations_A=survival_times[mask_low],
        durations_B=survival_times[mask_high],
        event_observed_A=events[mask_low],
        event_observed_B=events[mask_high]
    )

    return results.p_value


def save_risk_stratification_results(patient_ids, risk_scores, risk_groups, survival_times, events, save_path):
    """
    Save risk stratification results to CSV.

    Args:
        patient_ids: list of patient IDs
        risk_scores: np.array of risk scores
        risk_groups: np.array of group assignments
        survival_times: np.array of survival times
        events: np.array of event indicators
        save_path: str, path to save CSV
    """
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'risk_score': risk_scores,
        'risk_group': risk_groups,
        'risk_group_label': ['Low Risk' if g == 0 else 'High Risk' for g in risk_groups],
        'survival_time_days': survival_times,
        'event': events,
        'event_label': ['Censored' if e == 0 else 'Deceased' for e in events]
    })

    df = df.sort_values('risk_score', ascending=False)
    df.to_csv(save_path, index=False)
    print(f"[Survival Utils] Risk stratification results saved to {save_path}")


def compute_survival_metrics(survival_times, events, risk_groups):
    """
    Compute survival metrics for each risk group.

    Args:
        survival_times: np.array of survival times (days)
        events: np.array of event indicators (1=deceased, 0=censored)
        risk_groups: np.array of risk group assignments (0=low risk, 1=high risk)

    Returns:
        metrics: dict with median survival times and HR
    """
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    mask_low = (risk_groups == 0)
    mask_high = (risk_groups == 1)

    metrics = {}

    # Fit curves
    if mask_low.sum() > 0:
        kmf_low.fit(survival_times[mask_low], events[mask_low])
        metrics['median_survival_low_risk'] = kmf_low.median_survival_time_
    else:
        metrics['median_survival_low_risk'] = None

    if mask_high.sum() > 0:
        kmf_high.fit(survival_times[mask_high], events[mask_high])
        metrics['median_survival_high_risk'] = kmf_high.median_survival_time_
    else:
        metrics['median_survival_high_risk'] = None

    # Log-rank test
    metrics['logrank_p_value'] = compute_logrank_pvalue(survival_times, events, risk_groups)

    # Group sizes
    metrics['n_low_risk'] = mask_low.sum()
    metrics['n_high_risk'] = mask_high.sum()

    return metrics
