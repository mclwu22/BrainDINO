## ⚠️ Code & Weights Availability

The pretrained weights and full self-supervised pretraining pipeline are not released at this stage due to the large-scale multi-dataset integration and substantial computational requirements. All methodological details are described in the manuscript, and the full code and weights will be made publicly available upon publication.
# BrainDINO Downstream Evaluation

This directory provides an organized, task-oriented entry layer for the downstream evaluation suite used in the BrainDINO paper. The goal is not to replace every historical experiment script in the repository. Instead, it offers a compact, reproducible overview of the core downstream tasks, together with unified launch points for the main BrainDINO comparisons.

The organized runners in this folder are designed around the evaluation protocol used throughout the paper:

- BrainDINO is evaluated as a pretrained encoder on a diverse set of downstream neuroimaging tasks.
- Most supervised tasks use lightweight task heads on top of a frozen or fixed pretrained encoder backbone.
- Frozen-feature evaluation is explicitly included through a subject-level kNN pipeline on ADNI and OASIS.
- The same interface can be used to compare BrainDINO against DINOv3, BM-MAE, BrainMVP, and BrainIAC wherever the corresponding task code is available.

## What Is Included

The current organized task layer covers the following downstream settings:

- Classification: ABIDE, ADNI, OASIS, BraTS sequence classification, and binary survival classification on UPENN-GBM.
- Regression: brain age estimation on IXI + LONG579 + Pixar, and post-stroke temporal prediction on ATLAS.
- Mutation prediction: UCSF-PDGM multi-modal mutation classification.
- Survival risk modeling: multi-modal Cox-style survival modeling and Kaplan-Meier risk stratification on UPENN-GBM.
- Frozen encoder analysis: subject-level kNN probing on ADNI and OASIS, using cached frozen features from the paper visualization pipeline.
- Segmentation: a reserved placeholder is kept in the registry for future cleanup of the tumor segmentation entrypoints.

## Evaluation Protocol

### Supervised downstream tasks

For the supervised classification and regression tasks, the organized runners wrap the existing LUNA16 downstream trainers. In the default organized setup:

- BrainDINO corresponds to the `meddinov3` backbone.
- The default mode uses pretrained encoders with lightweight downstream heads.
- Classification and regression tasks are launched through unified task IDs, instead of separate one-off scripts.
- Mutation and survival retain their dedicated task-specific trainers, because those tasks require multi-modal inputs and custom objectives.

### Frozen-feature kNN probing

The frozen-feature kNN experiments are implemented in the repository-level path:

`Foundation_model_paper_contents/Visualization/`

This pipeline:

- loads pretrained encoders without downstream finetuning,
- extracts subject-level features from fixed train/test splits,
- caches those features under `feature_cache/`,
- evaluates kNN accuracy across multiple `k` values,
- and produces the paper-facing ADNI/OASIS figures and McNemar significance analyses.

Within the organized task registry, this experiment is exposed as `knn_adni_oasis`.

## Quick Start

Run everything from the LUNA16 root:

```bash
cd /home/exx/Desktop/Med_DINOv3/Finetune/Large-Scale-Medical/Downstream/monai/LUNA16
```

List the organized tasks:

```bash
python -m downstream_tasks.run --list
```

Run a supervised downstream task:

```bash
python -m downstream_tasks.run --task adni_cls --encoder meddinov3
python -m downstream_tasks.run --task brain_age_reg --encoder bm_mae --train-ratio 0.4
python -m downstream_tasks.run --task mutation_ucsf --encoder brainmvp
python -m downstream_tasks.run --task upenn_survival_cox --encoder brainiac
```

Run the frozen-feature kNN evaluation:

```bash
python -m downstream_tasks.run --task knn_adni_oasis --encoder meddinov3
```

Inspect a configuration without launching training or evaluation:

```bash
python -m downstream_tasks.run --task atlas_reg --dry-run
```

Launch a short sanity configuration:

```bash
python -m downstream_tasks.run --task adni_cls --quick
```

## Task Registry

The organized task mapping is defined in `registry.py`. It links each paper-facing task to:

- a stable task ID,
- its dataset and evaluation metric,
- the underlying training or evaluation entrypoint,
- and the corresponding paper table where applicable.

The main runnable tasks are:

| task_id | task type | dataset | paper-facing purpose |
|---|---|---|---|
| `abide_cls` | classification | ABIDE | neurological disorder classification |
| `adni_cls` | classification | ADNI | Alzheimer's disease classification |
| `oasis_cls` | classification | OASIS | cognitive impairment classification |
| `brats_sequence_cls` | classification | BraTS2023 aggregated cohort | MRI sequence classification |
| `upenn_survival_cls` | classification | UPENN-GBM | binary survival prediction |
| `brain_age_reg` | regression | IXI + LONG579 + Pixar | brain age estimation |
| `atlas_reg` | regression | ATLAS | post-stroke temporal prediction |
| `mutation_ucsf` | mutation | UCSF-PDGM | multi-modal mutation prediction |
| `upenn_survival_cox` | survival | UPENN-GBM | Kaplan-Meier risk stratification |
| `knn_adni_oasis` | frozen-feature analysis | ADNI + OASIS | subject-level kNN probing |

## Repository Structure

- `run.py`
  Single command-line entrypoint for the organized downstream task layer.

- `registry.py`
  Formal mapping from task IDs to datasets, metrics, code entrypoints, and paper tables.

- `runners/`
  Thin wrappers around the existing downstream training code. These are intentionally lightweight and avoid rewriting the original task logic.

- `Foundation_model_paper_contents/Visualization/`
  Frozen-feature kNN evaluation, feature caching, ADNI/OASIS plotting, and McNemar significance analysis.

## Segmentation

Tumor segmentation code is provided in the `BrainDINO_seg/` directory (see repository structure above).

This module follows an nnU-Net–based pipeline, adapted to support BrainDINO as a frozen or partially frozen encoder within the segmentation framework. Due to the complexity of the original training setup and dataset dependencies, the segmentation entrypoints are not yet unified into a single minimal launcher within the organized downstream task interface.

For detailed usage, configuration, and training procedures, please refer to the README file inside the `BrainDINO_seg/` folder.

We will further streamline and integrate these entrypoints into the main downstream task registry in future updates.

## Outputs

By default, organized runs are written to:

```text
LUNA16/downstream_tasks_runs/<task_id>/<encoder>/ratio_<N>/
```

Each organized run writes at least:

- `run_manifest.json`
- `summary.json`
- task-specific checkpoints, CSVs, or per-case predictions when the underlying runner produces them

## Scope

This directory is intended as the paper-facing downstream overview for the BrainDINO repository. It focuses on the core evaluation tasks and on a clean entry layer around the existing codebase. Historical sweeps, robustness scripts, plotting notebooks, and report-building utilities remain in their original locations and are not duplicated here.
