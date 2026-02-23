# TEE-RedBench (code for the paper)

This repository contains the evaluation code and prompt suite used for **TEE-RedBench**, described in the paper:

> *Red-Teaming Claude Opus and ChatGPT-based Security Advisors for Trusted Execution Environments*

The pipeline red-teams LLM assistants in the role of **TEE security advisors**, measures **rubric quality** and **prompt-induced failure modes**, and reports **failure transferability** and **defense ablation** results (Tables 5–7 in the paper).

---

## What’s in this repo

```
.
├── eval.py          # Main entrypoint (collect / annotate / analyze)
├── config.yaml          # Experiment configuration (models, sampling, rubric, defenses)
├── prompts.csv          # TEE-RedBench prompt suite (208 prompts, 6 families)
└── autorun.sh           # Convenience script to reproduce Tables 5–7
```


---

## Quickstart

### 1) Environment

- Python 3.10+ recommended
- API keys for the model providers you enable in `config.yaml`

Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

Set credentials (or put them in a `.env` file):

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

### 2) Run the full experiment (baseline + defenses + analysis)

```bash
bash autorun.sh
```

This will:

1) collect model outputs for each prompt/setting  
2) annotate outputs with a rubric + failure labels (via a judge model)  
3) analyze and emit CSV + LaTeX rows for Tables 5–7  

---

## Commands

### Collect model outputs

```bash
python eval.py --config config.yaml collect --setting baseline
```

Supported settings (match the paper’s Table 7 ablations):

- `baseline`
- `policy_gating`
- `retrieval_grounding`
- `structured_template`
- `verification_checks`
- `all_defenses`

### Annotate outputs (rubric + failures)

```bash
python eval.py --config config.yaml annotate --setting baseline
```

### Analyze and export tables

```bash
python eval.py --config config.yaml analyze
```

---

## Outputs and file formats

All outputs are written under `out_dir` (default: `./tee_eval_out`).

### 1) Raw runs (model answers)

`tee_eval_out/runs/<setting>/<eval_model>.jsonl`

Each JSONL record includes:
- prompt identifiers (`prompt_id`, `family`, `kind`, `paraphrase_id`, `sample_id`)
- `system_prompt` and the `final_user_prompt` shown to the model
- `answer`
- `latency_ms` and token usage (when returned by the provider SDK)
- a stable `run_key` for debugging/replay

### 2) Annotations (judge outputs)

`tee_eval_out/ann/<setting>/<eval_model>.jsonl`

Each JSONL record contains:
- `total_score_0_14`
- per-axis scores: `axis_accuracy`, `axis_completeness`, ...
- binary failure labels: `boundary_confusion`, `attestation_overclaim`, ...
- derived labels: `overconfident_error`, `hallucination`

### 3) Analysis artifacts (Tables 5–7)

Written to `tee_eval_out/results/`:
- `table5_results_family.csv` — mean rubric score by family (per model) + pooled risk rates
- `table6_transfer_matrix.csv` — failure transferability matrix (baseline)
- `table7_defense_ablation.csv` — prevalence + transferability by defense setting
- `tables_5_6_7.tex` — ready-to-paste LaTeX table rows

---

## Prompt suite (`prompts.csv`)

`prompts.csv` has the following columns:

- `prompt_id`: stable identifier (string)
- `family`: prompt family (e.g., Attestation & key mgmt)
- `kind`: prompt sub-type (free text)
- `base_prompt`: the canonical prompt text
- `paraphrases_json`: JSON array of paraphrases (string-encoded)

### Paraphrases / stress testing

By default, `paraphrases_json` is `[]` and the pipeline uses only `base_prompt`.

To enable paraphrase stress testing, either:
- populate `paraphrases_json` with a JSON list of paraphrase strings, or
- set `paraphrases.auto_generate_if_missing: true` in `config.yaml` to generate paraphrases on the fly.

---

## Configuration (`config.yaml`)

Key sections:

- `models`: which assistants are evaluated (e.g., chatgpt, claude) and their provider/model IDs
- `sampling`: K samples per prompt/paraphrase; temperature; `max_tokens`
- `judge`: judge model used to produce rubric scores + failure labels
- `rubric`: the 7 scoring axes (0–2 each, total 0–14)
- `failures`: the failure-mode taxonomy (includes core TEE failures + agentic/tool failures)
- `defenses`: toggles and prompts for the Table 7 ablations:
  - policy gating (system policy)
  - structured template (answer schema)
  - retrieval grounding (simple top‑k doc snippet injection from `docs/`)
  - verification checks (second-pass verifier rewrite)

---

## Retrieval grounding

If you use `retrieval_grounding` / `all_defenses`, add plaintext references to:

```
docs/
  vendor_docs.md
  advisories.txt
  ...
```

The pipeline will select `top_k` snippets (by simple token overlap) and prepend them as “REFERENCE CONTEXT”.

---

## Reproducibility Notes

- LLM outputs are stochastic: even with a fixed temperature, results may vary across time/provider updates.
- The provided pipeline can use both an LLM judge or human . For human annotations, we:
  1) ran `collect`,
  2) label outputs externally,
  3) produce annotation JSONLs with the same schema under `tee_eval_out/ann/`,
  4) then ran `analyze`.

---

## Citation

If you use this code/prompt suite in academic work, please cite the paper:

```bibtex
@misc{tee_redbench_2026,
  title        = {Red-Teaming Claude Opus and ChatGPT-based Security Advisors for Trusted Execution Environments},
  author       = {Anonymous Authors},
  year         = {2026},
  note         = {Anonymous submission to ACM CAIS 2026},
}
```

If you also want to cite the software artifact, you can add:

```bibtex
@misc{tee_redbench_code,
  title        = {TEE-RedBench: Code and prompt suite},
  author       = {Anonymous Authors},
  year         = {2026},
  howpublished = {\url{https://anonymous.4open.science/status/tee-redbench-xxxx}},
}
```

---

## License

MIT

---

## Security / responsible use

This repo contains policy-bound misuse probes intended to test refusal robustness. Do not use the prompts or outputs to develop or operationalize offensive capabilities. Please follow your organization’s responsible disclosure and red-teaming policies.
