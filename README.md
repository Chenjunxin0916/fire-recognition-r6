# Fire-recognition R6 source candidate

This is a **local release candidate**, not a published repository or a claim that the reported results have been independently reproduced. It contains source for the revised 1,100-image, 427-component internal study: the initial Standard/HN/Focal Sampling comparison and the later fixed random-negative replay control. The random control was designed after HN TEST results were inspected.

The main `src/` tree is based on the frozen source snapshot of the later random-control release. Six files that differed in the earlier formal three-arm execution are preserved under `formal_release_source/`; see `SOURCE_PROVENANCE.md`. The package retains licensing notices from the preceding public source archive while replacing its outdated study code. It is a curated post-experiment package, not a byte-identical Git commit of either historical run.

## What is included

- Core Python source, formal and later-control execution entrypoints, and the two accepted R6 configuration families.
- Analysis source for formal statistics, large-component sensitivity, cost reconciliation, and manuscript figures.
- Environment version records, source hashes, and third-party attribution.

## What is absent

No D-Fire image or label, study-specific selection/fold map, model checkpoint, image-level prediction, CAL threshold table, archived run record, credential, or private audit ledger is bundled. The public D-Fire collection alone cannot reconstruct the exact 1,100-image subset or revised fold mapping. See `DATA_AND_WEIGHTS.md`.

## Verification scope

Pure source syntax and selected synthetic code paths can be checked without study data. Recomputing reported metrics requires an authorized complete prediction table, matching CAL thresholds, the exact component/fold mapping, and the documented analysis environment. Independent retraining additionally requires lawful image access and the initial model assets. No full retraining or bootstrap rerun is claimed for this candidate.

The recorded environment was Windows, Python 3.11.9, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and Ultralytics 8.4.92. `requirements/` records the original dependency versions; it contains no packages or pretrained weights. `pyproject.toml` uses a candidate package version until an actual release is approved.

## Code map

| Path | Role |
| --- | --- |
| `src/fire_detection/research/` | FIT-only OOF, replay, Focal adaptation, CAL/TEST, paired analysis, and later random-control modules |
| `scripts/` | Frozen release preparation and execution entrypoints |
| `configs/split_v2_classifier_preflight_B_20261021_v2/` | Revised-split initial three-arm configuration |
| `configs/hn_random_control_split_v2_20261022_v2/` | Later fixed random-control configuration |
| `formal_release_source/` | Six formal-release source files that differ from the later snapshot |
| `analysis/` | Source for reported statistical, sensitivity, cost, and figure calculations; requires retained inputs |

The code is source-only under the recorded licence and third-party notices. Publication still requires rights review of the exact candidate tree. No new GitHub repository, release, or Zenodo record has been created by this local build.
