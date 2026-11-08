# PepFuse — code, data partitions and training logs

This bundle holds the code snapshot, data partitions and run logs behind the manuscript "PepFuse: integrating sequence, graph, and
descriptor representations for peptide property prediction" (Safari and Khotanlou).

## Contents

| Folder | What it holds |
|---|---|
| `code/` | The code snapshot from the first author's working folder. The fused models were trained with `model/train_fusion_enhanced.py` and the two runs marked `earlier_script` with `model/train_fusion.py`. The historical versions of these scripts were not preserved separately, so the snapshot may differ in details from the version used for a given run (see Known issues); `runs/<run>/config.json` and `console.log` record what each run actually did. |
| `code/config_fusion_handcrafted.yaml`, `code/yaml/` | Configuration templates. The exact settings of each run are in `runs/<run>/config.json`. |
| `partitions/hemo`, `partitions/nf` | The fixed train/validation/test partitions used by all runs (`inputs`: token ids, 0 = padding; `labels`: 1 = positive). Hemolysis has 7,545/839/932 sequences and Non-fouling has 13,919/1,547/1,719. The 45-dimensional descriptor features are not stored here; rebuild them with `code/scripts/add_classic_features.py`. |
| `runs/runs.csv` | One row per logged run, matching Supplementary Table S1. `in_means` says whether the run is included in the means reported in the paper. |
| `runs/<run>/config.json` | The configuration logged to Weights & Biases for that run. |
| `runs/<run>/console.log` | The run's console output with progress bars removed. It contains the per-epoch validation metrics and the test results. |

## Reading the test results

Each run evaluates two checkpoints on the test partition: the one with the highest validation F1-score and the one
with the highest validation accuracy. The block labelled "Best Test Results" is whichever of the two had the higher
**test** accuracy. The paper does not use that choice, because it depends on the test set. The paper reports the
best-validation-F1 checkpoint of every run:

- the block "F1 Model Test Results", or
- the block "Best Test Results" when the log says `Best Model Type: F1`, or
- the single test block in the two `earlier_script` runs.

## Known issues

- **Descriptor features.** `code/scripts/add_classic_features.py` maps token ids to amino acids with a table
  (`TOKEN_TO_AA`) that disagrees with the ProtBERT vocabulary used to encode the sequences
  (`code/data/convert_encodings.py`) for 11 of the 20 standard amino acids. Its BLOSUM62 matrix also has −1
  instead of 0 for A–C. The S+G+D runs were trained with these features. Correct both before reusing the
  descriptor branch.
- **Contrastive loss.** `fusion_hemo_1210_2201` and `fusion_nf_1210_1859` were trained with an added supervised
  contrastive loss (`use_supcon: true`), so they are excluded from the means in the paper.
- **Early stopping.** Included runs that stopped early reported a patience of 10 epochs; the released
  `train_fusion_enhanced.py` uses 8. Other runs reached their epoch limit.
- **One-hot flag.** `graph.use_one_hot` in the run configs is not read by the data loader.
- **Hemolysis duplicates.** 46.8% of the Hemolysis test observations (47.7% of the validation observations)
  have an identical sequence in the training partition.

## Environment

Python with PyTorch, PyTorch Geometric and HuggingFace Transformers. The runs used single 12 GB NVIDIA RTX 3060
GPUs. No random seed was set, so re-running gives slightly different results.

## Data source

The Hemolysis and Non-fouling datasets are those distributed with PeptideBERT (Guntuboina et al., 2023).
