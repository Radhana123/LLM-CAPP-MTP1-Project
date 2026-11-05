# Week 1 — Feature Extraction & Tokenization
## LLM-Based Intelligent Process Planning System for Manufacturing

### Objective
Manufacturing part ke features (Hole, Slot, material, tolerance)
ko structured Token IDs mein convert karna jo LLM directly
process planning ke liye use kar sake.

### Files
| File | Description |
|------|-------------|
| `feature_vocab.py` | 10 geometry features ki vocabulary |
| `material_tokens.py` | 19 materials (7 + 12 of Step 55) + tolerance + batch categories |
| `token_map.json` | Master Token ID mapping |
| `parser.py` | JSON input validator with error handling |
| `tokenizer.py` | Main: input → token sequence |
| `tests/test_tokenizer.py` | 6 pytest unit tests |

### Run
```bash
python tokenizer.py
python -m pytest tests/ -v
```

### Test Results
6 passed ✅

### Example
Input:
```json
{"material":"Aluminum","features":["Hole","Slot"],
 "tolerance":"0.02mm","batch_size":500}
```
Output:

### Week 2 Plan
Token sequences Week 2 mein LLM Process Planner ko diye jayenge.