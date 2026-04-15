RUNNER_TARGETS = {
    "classification": ("downstream_tasks.runners.classification", "run_task"),
    "regression": ("downstream_tasks.runners.regression", "run_task"),
    "mutation": ("downstream_tasks.runners.mutation", "run_task"),
    "survival_cox": ("downstream_tasks.runners.survival", "run_cox_task"),
    "knn": ("downstream_tasks.runners.knn", "run_task"),
}

__all__ = ["RUNNER_TARGETS"]
