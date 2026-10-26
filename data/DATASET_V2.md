# Dataset v2 (MTP-1)

Synthetic manufacturing parts for LLM-CAPP. Regenerate with `python scripts/make_dataset_v2.py` (fixed seeds, output is deterministic).

| File | Parts | Purpose |
|---|---|---|
| `dev_v2.json` / `.csv` | 200 | Design and tuning. Also the retrieval pool for few-shot examples. |
| `test_v2.json` / `.csv` | 120 | Sealed. Final runs only. 60 parts from dev archetypes + 60 from 7 held-out archetypes (feature sets never seen in dev). |

Per part: `material`, `features` (36-feature vocabulary), `tolerance`, `batch_size`, `dims` (diameter, length, depth, width in mm),
`required_ra_um` (from tolerance), `machine_preference`, `archetype`, `in_distribution`, plus derived `n_features`, `batch_category`,
`machine_profile` (lathe / milling / mixed / flexible). Optional `reference_routes` is reserved for the expert benchmark.

Dev composition: 36 archetypes (5 or 6 parts each), 36/36 features, 7 materials, 6 tolerances, 4 batch categories,
2 to 8 features per part (18 parts with 7 or 8 features).

Known limits: archetypes are synthetic and not expert-validated; dims are uniform random in family ranges; Plastic and Copper are rare;
the tolerance to Ra mapping is a rule of thumb (ISO 1302 preferred values).
