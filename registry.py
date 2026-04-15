from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    category: str
    status: str
    paper_task: str
    dataset: str
    metric: str
    paper_tables: tuple[str, ...]
    runner_name: str | None
    code_entry: str | None
    dataset_name: str | None = None
    supported_encoders: tuple[str, ...] = ()
    notes: str = ""


_GENERAL_ENCODERS = ("meddinov3", "dinov3", "brainmvp", "bm_mae", "brainiac", "scratch")


TASK_SPECS = {
    "abide_cls": TaskSpec(
        task_id="abide_cls",
        category="classification",
        status="runnable",
        paper_task="Neurological Disorder Classification",
        dataset="ABIDE",
        metric="Macro-AUC",
        paper_tables=("table_neurodevelopmental_neurodegenerative_conditions_classification.tex",),
        runner_name="classification",
        code_entry="trainers/dinov3_volume2d_trainer_general.py",
        dataset_name="ABIDE",
        supported_encoders=_GENERAL_ENCODERS,
        notes="Matches the paper ABIDE classification task.",
    ),
    "adni_cls": TaskSpec(
        task_id="adni_cls",
        category="classification",
        status="runnable",
        paper_task="Alzheimer's Disease Classification",
        dataset="ADNI",
        metric="Macro-AUC",
        paper_tables=(
            "table_neurodevelopmental_neurodegenerative_conditions_classification.tex",
            "table_adni_age_stratified_best_competitor.tex",
        ),
        runner_name="classification",
        code_entry="trainers/dinov3_volume2d_trainer_general.py",
        dataset_name="ADNI",
        supported_encoders=_GENERAL_ENCODERS,
        notes="Age-stratified ADNI analysis should reuse predictions from this task.",
    ),
    "oasis_cls": TaskSpec(
        task_id="oasis_cls",
        category="classification",
        status="runnable",
        paper_task="Neurological Disorder Classification",
        dataset="OASIS",
        metric="Macro-AUC",
        paper_tables=("table_neurodevelopmental_neurodegenerative_conditions_classification.tex",),
        runner_name="classification",
        code_entry="trainers/dinov3_volume2d_trainer_general.py",
        dataset_name="OASIS",
        supported_encoders=_GENERAL_ENCODERS,
        notes="Uses the simplest train.csv / valid.csv setup instead of the larger OASIS_ATLAS sweep script.",
    ),
    "brats_sequence_cls": TaskSpec(
        task_id="brats_sequence_cls",
        category="classification",
        status="runnable",
        paper_task="MRI Sequence Classification",
        dataset="BraTS2023 aggregated cohort",
        metric="Macro-AUC",
        paper_tables=("table_sequence_classification.tex",),
        runner_name="classification",
        code_entry="trainers/dinov3_volume2d_trainer_general.py",
        dataset_name="BraTS",
        supported_encoders=_GENERAL_ENCODERS,
        notes="Maps to the BraTS modality-classification task used in the paper table.",
    ),
    "upenn_survival_cls": TaskSpec(
        task_id="upenn_survival_cls",
        category="classification",
        status="runnable",
        paper_task="Binary Survival Prediction",
        dataset="UPENN-GBM",
        metric="Macro-AUC",
        paper_tables=("table_survival_prediction.tex",),
        runner_name="classification",
        code_entry="trainers/dinov3_volume2d_trainer_general.py",
        dataset_name="UPENN",
        supported_encoders=_GENERAL_ENCODERS,
        notes="This is the paper's binary survival classification task, not the Cox/KM task.",
    ),
    "brain_age_reg": TaskSpec(
        task_id="brain_age_reg",
        category="regression",
        status="runnable",
        paper_task="Brain Age Regression",
        dataset="IXI + LONG579 + Pixar",
        metric="MAE",
        paper_tables=(
            "table_brain_age_regression.tex",
            "table_brain_age_age_stratified_best_competitor.tex",
        ),
        runner_name="regression",
        code_entry="trainers/dinov3_volume2d_trainer_general_regression.py",
        dataset_name="Combine_IXI_LONG_PIXAR",
        supported_encoders=_GENERAL_ENCODERS,
        notes="Age-stratified brain-age analysis should reuse predictions from this task.",
    ),
    "atlas_reg": TaskSpec(
        task_id="atlas_reg",
        category="regression",
        status="runnable",
        paper_task="Post-Stroke Temporal Prediction",
        dataset="ATLAS",
        metric="MAE",
        paper_tables=("table_post_stroke_regression.tex",),
        runner_name="regression",
        code_entry="trainers/dinov3_volume2d_trainer_general_regression.py",
        dataset_name="ATLAS",
        supported_encoders=_GENERAL_ENCODERS,
        notes="Uses the plain ATLAS train.csv / valid.csv setup for a minimal runnable entry.",
    ),
    "mutation_ucsf": TaskSpec(
        task_id="mutation_ucsf",
        category="mutation",
        status="runnable",
        paper_task="Mutation Prediction",
        dataset="UCSF-PDGM",
        metric="Macro-AUC",
        paper_tables=("table_mutation_prediction.tex",),
        runner_name="mutation",
        code_entry="trainers/mutation_ucsf_trainer.py",
        supported_encoders=_GENERAL_ENCODERS,
        notes="Two-modality T1c + FLAIR mutation classification.",
    ),
    "upenn_survival_cox": TaskSpec(
        task_id="upenn_survival_cox",
        category="survival",
        status="runnable",
        paper_task="Kaplan-Meier Risk Stratification",
        dataset="UPENN-GBM",
        metric="Log-rank p-value / median survival",
        paper_tables=("table_survival_prediction.tex",),
        runner_name="survival_cox",
        code_entry="SurvivalRiskStratification_KaplanMeier/trainers/survival_multimodal_trainer.py",
        supported_encoders=_GENERAL_ENCODERS,
        notes="Multi-modal Cox-style survival runner with Kaplan-Meier output.",
    ),
    "knn_adni_oasis": TaskSpec(
        task_id="knn_adni_oasis",
        category="analysis",
        status="runnable",
        paper_task="Frozen-Feature kNN Probing",
        dataset="ADNI + OASIS",
        metric="Accuracy",
        paper_tables=("table_knn_adni_oasis_exact_mcnemar.tex",),
        runner_name="knn",
        code_entry="Foundation_model_paper_contents/Visualization/run_knn_eval.py",
        supported_encoders=("meddinov3", "dinov3", "brainmvp", "bm_mae", "brainiac"),
        notes="Uses the frozen-feature kNN pipeline under Foundation_model_paper_contents/Visualization.",
    ),
    "tumor_seg_brats2023_mets": TaskSpec(
        task_id="tumor_seg_brats2023_mets",
        category="segmentation",
        status="placeholder",
        paper_task="Tumor Segmentation",
        dataset="BraTS2023-Mets",
        metric="Dice",
        paper_tables=("table_tumor_segmentation.tex",),
        runner_name=None,
        code_entry=None,
        notes="Paper result exists, but this LUNA16 folder does not contain a clean BraTS tumor-seg training entry for this dataset.",
    ),
    "tumor_seg_brats2023_men": TaskSpec(
        task_id="tumor_seg_brats2023_men",
        category="segmentation",
        status="placeholder",
        paper_task="Tumor Segmentation",
        dataset="BraTS2023-MEN",
        metric="Dice",
        paper_tables=("table_tumor_segmentation.tex",),
        runner_name=None,
        code_entry=None,
        notes="Paper result exists, but this LUNA16 folder does not contain a clean BraTS tumor-seg training entry for this dataset.",
    ),
    "tumor_seg_brats2024_goat": TaskSpec(
        task_id="tumor_seg_brats2024_goat",
        category="segmentation",
        status="placeholder",
        paper_task="Tumor Segmentation",
        dataset="BraTS2024-GoAT",
        metric="Dice",
        paper_tables=("table_tumor_segmentation.tex",),
        runner_name=None,
        code_entry=None,
        notes="Paper result exists, but this LUNA16 folder does not contain a clean BraTS tumor-seg training entry for this dataset.",
    ),
    "tumor_seg_brats2021": TaskSpec(
        task_id="tumor_seg_brats2021",
        category="segmentation",
        status="placeholder",
        paper_task="Tumor Segmentation",
        dataset="BraTS2021",
        metric="Dice",
        paper_tables=("table_tumor_segmentation.tex",),
        runner_name=None,
        code_entry=None,
        notes="Paper result exists, but this LUNA16 folder does not contain a clean BraTS tumor-seg training entry for this dataset.",
    ),
}


def get_task_spec(task_id: str) -> TaskSpec:
    if task_id not in TASK_SPECS:
        raise KeyError(f"Unknown task '{task_id}'.")
    return TASK_SPECS[task_id]


def iter_task_specs():
    for task_id in sorted(TASK_SPECS):
        yield TASK_SPECS[task_id]
