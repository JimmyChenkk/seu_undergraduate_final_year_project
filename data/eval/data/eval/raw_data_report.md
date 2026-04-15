# TE Raw Data Inspection

Generated at: `2026-04-15T05:50:13.095976+00:00`
Raw dir: `data/raw`

## File Summary

| File | Domain | Type | Top-level keys | Error |
| --- | --- | --- | --- | --- |

## Quick takeaways

- Signals shape variants: `N/A`
- Fold names observed: `N/A`
- Label preview frequency (not class counts): `N/A`
- Next step: inspect domain-wise sample counts, class balance, and signal normalization strategy.

## Mermaid overview

```mermaid
flowchart TD
    A[Raw pickle files in data/raw] --> B[pickle.load(handle)]
    B --> C{Top-level dict?}
    C -->|Yes| D[Signals array]
    C -->|Yes| E[Labels array]
    C -->|Yes| F[Folds dict]
    D --> G[Benchmark manifest / dataset loader]
    E --> G
    F --> G
    G --> H[Normalization / split selection]
    H --> I[Single-source or multi-source batches]
    I --> J[Training methods: source_only / coral / dan / dann / cdan / deepjdot / rcta]
```
