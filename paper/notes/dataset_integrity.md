# Dataset Integrity Check

- Dataset: `te_domain_adaptation`
- Manifest: `data/benchmark/manifest.json`
- Domain count: `6`
- Channel contract: 34 continuous variables interpreted as XME(1)-XME(22) + XMV(1)-XMV(12)
- Window contract: 600 time steps per sample; loader transposes to 34 x 600 when channels_first=true
- Normalization: `standardization` / scope `domain`
- Default target labels enabled: `False`
- Overall status: `pass`

## Domains

| Domain | Signals | Labels | Label Range | Classes | Folds | Status |
| --- | --- | --- | --- | --- | --- | --- |
| mode1 | [2900, 600, 34] | [2900] | {'min': 0, 'max': 28} | 29 | 5 | ok |
| mode2 | [2845, 600, 34] | [2845] | {'min': 0, 'max': 28} | 29 | 5 | ok |
| mode3 | [2899, 600, 34] | [2899] | {'min': 0, 'max': 28} | 29 | 5 | ok |
| mode4 | [2865, 600, 34] | [2865] | {'min': 0, 'max': 28} | 29 | 5 | ok |
| mode5 | [2883, 600, 34] | [2883] | {'min': 0, 'max': 28} | 29 | 5 | ok |
| mode6 | [2897, 600, 34] | [2897] | {'min': 0, 'max': 28} | 29 | 5 | ok |

## Warnings

- mode2 fold lengths sum to 2835, not the full sample count 2845; current loader treats the selected fold as eval and all remaining samples as train.
- mode3 fold lengths sum to 2895, not the full sample count 2899; current loader treats the selected fold as eval and all remaining samples as train.
- mode4 fold lengths sum to 2855, not the full sample count 2865; current loader treats the selected fold as eval and all remaining samples as train.
- mode5 fold lengths sum to 2880, not the full sample count 2883; current loader treats the selected fold as eval and all remaining samples as train.
- mode6 fold lengths sum to 2895, not the full sample count 2897; current loader treats the selected fold as eval and all remaining samples as train.
- normalization_scope is domain/full-domain: this matches the local benchmark but uses full target-domain statistics, including the held-out fold.
