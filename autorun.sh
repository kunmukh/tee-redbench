export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# 1) Baseline (Table 5 + Table 6 baseline + also baseline row for Table 7)
python run_eval.py --config config.yaml collect  --setting baseline
python run_eval.py --config config.yaml annotate --setting baseline

# 2) Defense ablations (Table 7)
for s in policy_gating retrieval_grounding structured_template verification_checks all_defenses; do
  python run_eval.py --config config.yaml collect  --setting $s
  python run_eval.py --config config.yaml annotate --setting $s
done

# 3) Analyze + emit LaTeX rows
python run_eval.py --config config.yaml analyze