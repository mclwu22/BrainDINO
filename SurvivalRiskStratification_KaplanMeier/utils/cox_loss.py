"""
Cox Partial Likelihood Loss for Survival Analysis

Reference:
- Cox, D. R. (1972). Regression models and life-tables.
- Katzman et al. (2018). DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.
"""

import torch
import torch.nn as nn


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards Loss.

    Handles right-censored data by computing partial likelihood over risk sets.

    Args:
        risk_scores: (B,) - predicted continuous risk scores (higher = higher risk)
        survival_times: (B,) - observed survival times T
        events: (B,) - event indicators E (1=death, 0=censored)

    Returns:
        Negative log partial likelihood (scalar loss)

    How it works:
    - For each uncensored sample i (E_i=1):
        - Find all samples j still at risk at time T_i (T_j >= T_i)
        - Compute: log P(i dies | risk set) = log_risk_i - log(sum_j exp(log_risk_j))
    - Sum over all uncensored samples
    - Censored samples only contribute to risk sets, not as "events"
    """

    def __init__(self):
        super().__init__()

    def forward(self, risk_scores, survival_times, events):
        """
        Compute Cox partial likelihood loss.

        Args:
            risk_scores: (B,) tensor of predicted risk scores
            survival_times: (B,) tensor of survival times
            events: (B,) tensor of event indicators (1=death, 0=censored)

        Returns:
            Scalar loss (negative log partial likelihood)
        """
        # Ensure correct shapes
        risk_scores = risk_scores.view(-1)
        survival_times = survival_times.view(-1)
        events = events.view(-1)

        # Sort by survival time (descending) for efficient risk set computation
        # Ties in survival time are handled by PyTorch's stable sort
        sort_idx = torch.argsort(survival_times, descending=True)

        risk_scores_sorted = risk_scores[sort_idx]
        events_sorted = events[sort_idx]

        # Compute log of cumulative sum of exp(risk) for risk sets
        # risk_set[i] = log(sum_{j>=i} exp(risk_j))
        # This represents the denominator: all subjects at risk at time T_i
        log_risk_set = torch.logcumsumexp(risk_scores_sorted, dim=0)

        # Compute log partial likelihood for uncensored samples
        # log P(i | risk set) = risk_i - log(sum exp(risk_j))
        log_likelihood = (risk_scores_sorted - log_risk_set) * events_sorted

        # Average over uncensored samples (events == 1)
        # Negative because we want to maximize likelihood = minimize negative log likelihood
        num_events = events_sorted.sum()

        if num_events == 0:
            # Edge case: no events in batch (all censored)
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

        loss = -log_likelihood.sum() / num_events

        return loss


def concordance_index(risk_scores, survival_times, events):
    """
    Compute Harrell's C-index (concordance index) for survival analysis.

    C-index measures the fraction of all pairs of subjects whose predicted
    survival times are correctly ordered among all subjects that can actually be ordered.

    Args:
        risk_scores: (N,) predicted risk scores (higher = higher risk = shorter survival)
        survival_times: (N,) observed survival times
        events: (N,) event indicators (1=death, 0=censored)

    Returns:
        c_index: float in [0, 1], where 0.5 = random, 1.0 = perfect

    Definition:
    - A pair (i,j) is comparable if:
        - i had event (E_i=1) and T_i < T_j  (i died earlier)
    - Pair is concordant if: risk_i > risk_j  (higher risk → earlier death)
    - C-index = # concordant pairs / # comparable pairs
    """
    risk_scores = risk_scores.cpu().numpy() if torch.is_tensor(risk_scores) else risk_scores
    survival_times = survival_times.cpu().numpy() if torch.is_tensor(survival_times) else survival_times
    events = events.cpu().numpy() if torch.is_tensor(events) else events

    n = len(risk_scores)
    concordant = 0
    comparable = 0

    for i in range(n):
        if events[i] == 0:  # Censored, cannot be reference
            continue

        for j in range(n):
            if i == j:
                continue

            # i had event and died earlier than j's observed time
            if survival_times[i] < survival_times[j]:
                comparable += 1

                # Concordant if higher risk predicts earlier death
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5  # Tie

    if comparable == 0:
        return 0.5  # No comparable pairs

    return concordant / comparable


if __name__ == "__main__":
    # Test Cox loss
    print("Testing Cox PH Loss...")

    # Synthetic data
    risk_scores = torch.tensor([0.5, 1.2, 0.8, 2.0, 0.3])
    survival_times = torch.tensor([100.0, 50.0, 200.0, 30.0, 150.0])
    events = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])  # Sample 2 is censored

    criterion = CoxPHLoss()
    loss = criterion(risk_scores, survival_times, events)

    print(f"Risk scores: {risk_scores}")
    print(f"Survival times: {survival_times}")
    print(f"Events: {events}")
    print(f"Cox loss: {loss.item():.4f}")

    # Test C-index
    c_index = concordance_index(risk_scores, survival_times, events)
    print(f"C-index: {c_index:.4f}")

    print("\n✓ Cox loss implementation test passed!")
