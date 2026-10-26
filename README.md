# LLM-CAPP: An Agentic, LLM-Based Intelligent Process Planning System for Manufacturing

### Dynamic Route Construction · Machine-Aware Optimization · Multi-Objective Scoring · Interactive Analytics

**Summer Internship Project** · Manufacturing AI · CAPP (Computer-Aided Process Planning)
Department of Mechanical Engineering, IIT Kharagpur · Supervised by Prof. Sankha Deb, SRIC

---

## 📌 Project Status

**Weeks 1–6 Complete** | 58/58 Tests Passing | Interactive UI Live

| Week   | Module                                                  | Status      |
| ------ | -------------------------------------------------------- | ----------- |
| Week 1 | Feature Vocabulary, Tokenization, Dynamic Route Builder   | ✅ Complete |
| Week 2 | LLM Process Planner, Route Registry                       | ✅ Complete |
| Week 3 | Multi-Agent Evaluation (Time/Cost/Energy)                  | ✅ Complete |
| Week 4 | NSGA-II Optimization & FSM Validation                      | ✅ Complete |
| Week 5 | Error Detection & Self-Correction Loop                     | ✅ Complete |
| Week 6 | Full Pipeline, Interactive Streamlit UI                    | ✅ Complete |

---

## 🎯 Objective

Manual process planning in manufacturing is time-consuming, experience-dependent, and prone to sequencing errors. This project builds an AI-driven system that **automatically constructs, evaluates, validates, and self-corrects** manufacturing process plans — converting a raw part specification (material, geometric features, tolerance, batch size) into an optimized, machine-ready process route, presented through a live interactive dashboard.

Given a part's features, the system:

1. **Extracts** machining features automatically from a 2D engineering drawing or sketch using a Vision Language Model (VLM), or accepts manual feature selection.
2. **Constructs** a complete, precedence-valid manufacturing route (guaranteed to cover every requested feature).
3. **Groups operations by machine** (Lathe/Milling) to minimize costly changeovers.
4. **Scores** every candidate route on Time, Cost (₹), and Energy using dimension- and material-aware physical formulas.
5. **Optimizes** across these objectives with NSGA-II to surface the best trade-off — not just one arbitrary answer.
6. **Validates** every route (rule-based or LLM-generated) against a shared precedence rule set, self-correcting automatically on failure.
7. **Presents** the result on an interactive dashboard, with full traceability for every number shown.

---

## 🏗️ System Architecture

```
  Part Input (2D Drawing Image OR Manual Feature Selection)
        │
        ▼
┌─────────────────────┐
│  VLM Feature          │  Qwen3.8-27B Vision (via Groq) extracts machining
│  Extractor            │  features directly from uploaded engineering drawings
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│  Tokenizer            │  Converts input into a numeric token sequence
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│  Dynamic Route Builder│  Feature→Operations mapping + Precedence Graph
│  (+ Machine Grouping) │  → complete, valid, machine-grouped candidate routes
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│  Multi-Agent Scoring  │  Time / Cost (₹) / Energy / Efficiency agents,
│                        │  dimension- and material-aware
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│  NSGA-II Optimization │  Pareto-optimal route selection across objectives
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│  FSM Validator         │  Checks every candidate (rule-based or LLM) against
│  + Self-Correction     │  the shared precedence graph; repairs failures
└─────────┬─────────────┘
          ▼
   Interactive Dashboard (Streamlit)
   Route flow · Time/Cost breakdown · Pareto front · Feature mapping
```

An **LLM Planner** (Qwen3.8-27B via Groq) runs alongside the Route Builder as an additional candidate-route source, generating a plan directly from tokenized features. Its output passes through the identical FSM validation and self-correction path as every rule-based candidate — it is never trusted without verification.

### Core Design Principles

- **Image-driven feature extraction** — a VLM reads the uploaded 2D drawing and identifies machining features automatically; the user can review and edit before running the pipeline.
- **Completeness by construction** — a route is built feature-by-feature from a Feature-to-Operations mapping and a precedence graph, so a requested feature can never be silently omitted.
- **Machine-aware by default** — operations are grouped onto a single machine wherever the feature set allows, and a changeover is introduced only when genuinely unavoidable.
- **Physically grounded costing** — time and cost are derived from real manufacturing formulas (cutting speed, feed rate, multi-pass depth) using the part's actual dimensions and material, not fixed lookups.
- **No hidden numbers** — every total (time, cost) is decomposed into its components, and every tool/position change is individually traceable to its cause.
- **Trade-offs shown, not hidden** — multi-objective optimization surfaces the full set of good routes (the Pareto front), letting the user decide what matters most for a given job.
- **AI is verified, not trusted** — an LLM-generated route is checked against the exact same correctness rules as every rule-based route before it is ever used.

---

## 🖼️ Image-Based Feature Extraction

The most distinctive user-facing feature of the system: instead of manually selecting operations from a dropdown, the user uploads a 2D engineering drawing (PNG, JPG, or sketch photo) and the system extracts machining features automatically.

### How It Works

1. The uploaded image is sent to **Qwen3.8-27B Vision** (via Groq API) with a structured prompt listing all 36 valid feature names and asking the model to identify which are present.
2. The model returns a structured JSON response listing detected features, a confidence level, and notes explaining what it saw in the drawing.
3. The response is parsed and validated against the feature vocabulary — unrecognized terms are flagged and shown to the user separately.
4. Detected features are pre-filled into a **confirmation multiselect** — the user can add, remove, or correct before running the pipeline.

### What It Can Read

The extractor is designed to work on a range of drawing types:
- Formal dimensioned engineering drawings (ISO/ASME standard)
- CAD-generated 2D views
- Hand sketches or photos showing visible machining features (holes, slots, tapers, keyways, etc.)

Dimensions are **not required** — the extractor identifies feature types from geometry and annotation, not from numeric values.

---

## 🗺️ Route Only Mode — How It Works

This is the primary interface mode: a fast, dimension-free view of the recommended route and its trade-offs.

### Step 1 — User Input

Via the sidebar, the user: uploads a **2D engineering drawing** (AI extracts features automatically) or selects features manually. Then selects **Material**, **Tolerance**, **Batch Size**, and **Machine Preference** (Auto / Prefer Lathe / Prefer Milling), then clicks **Generate Process Plan**.

### Step 2 — Tokenization

The specification is converted into a fixed token sequence by dictionary lookup against `token_map.json`. Malformed input is rejected here, before anything downstream runs.

### Step 3 — Dynamic Route Construction

For the requested feature set, `route_builder.py` computes the required operations (merging overlapping requirements across features) and produces one or more valid orderings via a randomized topological sort — every candidate is complete and precedence-valid by construction.

### Step 4 — Machine-Aware Grouping

Each required operation is classified Lathe-only, Milling-only, or Shared. Shared operations are assigned to minimize machine changeovers (or steered by the user's preference); a route collapses onto a single machine whenever the feature set allows it, and a changeover marker — with its own time/cost/energy penalty — is inserted only when genuinely unavoidable.

### Step 5 — Multi-Agent Scoring (Reference Dimensions)

Each candidate route is scored by four agents — Time, Cost (₹), Energy, and a composite Efficiency score — using reference part dimensions (Ø25 mm, L = 50 mm, depth = 20 mm) so a route can be evaluated before exact dimensions are known.

### Step 6 — NSGA-II Pareto Selection

Where multiple valid routes exist, NSGA-II compares them by dominance, computes the non-dominated (Pareto) front, and selects the best overall trade-off, while the full front remains visible to the user.

### Step 7 — FSM Validation

The selected route is checked against the shared precedence rule set used to construct it, as an independent safety net.

### Step 8 — Dashboard Output

- **Color-coded route flow** — amber (Lathe), blue (Milling), purple (Shared), dark (Facing/Inspection), dashed amber (Machine Changeover).
- **Time Breakdown** — five metrics (Machining, Tool Change, Position Change, Changeover, Total) with a bar chart.
- **"Where changes occurred"** — an expandable, line-by-line trace of every tool change and position change and its cause.
- **Pareto Front** — interactive 3D (Time × Cost × Energy) and 2D (Time vs Cost, bubble = Energy) charts, with the selected route highlighted.
- **Completeness verification** — explicit confirmation that every requested feature is covered.
- **Feature → Operation mapping** — which operation(s) satisfy each requested feature, and on which machine.

---

## 🤖 LLM Integration — Where and How AI Is Used

Two distinct AI components are used in this system:

### 1. VLM Feature Extractor (Image Input)

**Model:** Qwen3.8-27B Vision via Groq API
**Role:** Reads an uploaded 2D engineering drawing and identifies which of the 36 machining features are present.
**File:** `week1/image_feature_extractor.py`

### 2. LLM Process Planner (Route Generation)

**Model:** Qwen3.8-27B via Groq API
**Role:** Generates a candidate process route directly from tokenized features, as a second independent path alongside the rule-based Route Builder.
**File:** `week2/llm_planner.py`

### Where the LLM Planner Sits in the Pipeline

```
Tokenized Features
        │
        ├──────────────────────────┐
        ▼                          ▼
Dynamic Route Builder      LLM Planner (Qwen3.8-27B via Groq)
(rule-based construction)  (generates a candidate route directly)
        │                          │
        └────────────┬─────────────┘
                      ▼
         FSM Validator (same 44-rule check for BOTH)
                      ▼
         Self-Correction (if invalid) → Route Builder called directly
                      ▼
              Multi-Agent Scoring → NSGA-II → Output
```

### What the Model Is Given

```
System Prompt (simplified):
"You are a manufacturing process planning expert.
 RULES: 1. Every plan MUST start with Facing, end with Inspection
        2. Use ONLY these valid operations: [full 41-operation list]
        3. Respect machining precedence
        4. Output ONLY a JSON array, nothing else"

User Prompt:
"Material: Steel, Features: Thread, Fillet, Tolerance: 0.02mm, Batch: 500"

Model Output:
["Facing", "External Threading", "Corner Rounding/Filleting", "Inspection"]
```

### Model Details

| Aspect | Detail |
| --- | --- |
| Model | Qwen3.8-27B (Vision + Text) |
| Hosting | Cloud API (Groq) — no local GPU required |
| Response time | ~1–3 seconds per request |
| Output format | Structured JSON, reliably parseable |
| Trust level | None by default — every output is validated before use |

### Where in the Code

| File | Role |
| --- | --- |
| `week1/image_feature_extractor.py` | VLM-based feature extraction from 2D drawings |
| `week2/llm_planner.py` | Builds the prompt, calls the Groq API, parses the JSON response |
| `week4/fsm_validator.py` | Validates the LLM's route using the shared precedence graph |
| `week5/self_corrector.py` | Requests a Route-Builder replacement if the LLM's route is invalid |

---

## 📂 Repository Structure

```
INTERN-PROJECT/
│
├── week1/                       Feature Vocabulary, Tokenization, Dynamic Route Builder
│   ├── feature_vocab.py         36 features, Feature→Operations mapping
│   ├── image_feature_extractor.py  VLM-based feature extraction from 2D drawings
│   ├── material_tokens.py       7 materials, tolerance & batch categories
│   ├── token_map.json           Token ID mapping (features, materials, 41 operations)
│   ├── parser.py                JSON input validator
│   ├── tokenizer.py             Input → token sequence
│   ├── precedence_graph.py      Operation-ordering rules, cycle-checked
│   ├── route_builder.py         Dynamic, machine-aware route construction
│   └── tests/                   Unit tests
│
├── week2/                       LLM Planner & Route Registry
│   ├── llm_planner.py           Groq / Qwen3.8-27B process-plan generation
│   ├── routes.py                Atomic operations registry
│   └── planner.py               End-to-end single-part planning
│
├── week3/                       Multi-Agent Evaluation
│   ├── agents.py                Time / Cost (₹) / Energy / Efficiency agents
│   ├── multi_agent_eval.py      Feature-aware route comparison
│   └── run_pipeline.py          Batch pipeline runner
│
├── week4/                       Optimization & Validation
│   ├── nsga2.py                 NSGA-II multi-objective optimization
│   ├── fsm_validator.py         Precedence validation (shared rule set)
│   └── week4_pipeline.py
│
├── week5/                       Self-Correction
│   ├── error_detector.py
│   ├── self_corrector.py        Route-Builder-backed guaranteed correction
│   └── week5_pipeline.py
│
├── week6/                       Full Pipeline & Interactive UI
│   ├── full_pipeline.py         Complete Week 1→5 pipeline
│   ├── final_report.py
│   └── app.py                   Streamlit interactive dashboard
│
├── data/                        Dataset
│   ├── generate_dataset.py      200 parts across 22 manufacturing archetypes
│   ├── parts_dataset.json/csv
│   └── final_results.json
│
└── README.md                    This file
```

---

## ⚙️ How to Run

**Set environment variables and launch the dashboard:**

```powershell
cd week6
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
streamlit run app.py
```

**Run the full pipeline on the dataset:**

```powershell
cd week6
python full_pipeline.py
```

**Run each module individually:**

```powershell
cd week1 && python route_builder.py
cd week4 && python nsga2.py
cd week5 && python self_corrector.py
```

**Run all tests:**

```powershell
cd week1 && python -m pytest tests/ -v
```

---

## 🔬 Test Summary

| Module                           | Tests | Status         |
| ---------------------------------- | ----- | -------------- |
| Week 1 — Vocabulary & Tokenizer     | 21    | ✅ All Passing |
| Week 1 — Route Builder              | 5     | ✅ All Passing |
| Week 3 — Multi-Agent Eval           | 10    | ✅ All Passing |
| Week 4 — NSGA-II & FSM              | 10    | ✅ All Passing |
| Week 5 — Self-Correction            | 12    | ✅ All Passing |
| **Total**                           | **58**| **✅ All Passing** |

---

## 📊 System at a Glance

| Metric                     | Value                                             |
| ---------------------------- | -------------------------------------------------- |
| Geometric features            | 36                                                   |
| Machining operations          | 41 (canonical, shared across all modules)            |
| Precedence rules enforced      | 44, cycle-checked                                    |
| Dataset                        | 200 parts across 22 realistic manufacturing archetypes |
| Optimization objectives        | 3 — Time, Cost, Energy                               |
| Currency                       | INR (1 USD = ₹95.595)                                |
| LLM / VLM                      | Qwen3.8-27B via Groq (~1–3s response time)           |
| Interface                      | Interactive Streamlit dashboard with Pareto visualization |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLM / VLM:** Qwen3.8-27B via Groq API (text + vision)
- **Optimization:** NSGA-II (custom implementation)
- **UI:** Streamlit, Plotly (interactive 3D/2D charts)
- **Testing:** pytest
- **Data:** JSON, CSV
- **Version Control:** Git & GitHub

---

## 🔜 Next Steps

- Calibrate cutting-parameter tables against real machining data where available.
- Review low-confidence precedence rules for possible promotion to enforced status.
- Extend the LLM Planner to accept free-text natural-language part descriptions.
- Add multi-sample consensus logic to the VLM feature extractor for higher extraction accuracy.

---

## 🔗 Repository

[github.com/Radhana123/LLM-CAPP-Project_Summer_Intern](https://github.com/Radhana123/LLM-CAPP-Project_Summer_Intern)