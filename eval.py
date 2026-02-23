import os, json, time, math, argparse, hashlib
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Utils
# ---------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def read_prompts(prompts_csv: str) -> pd.DataFrame:
    df = pd.read_csv(prompts_csv)
    required = {"prompt_id","family","kind","base_prompt","paraphrases_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prompts.csv missing columns: {missing}")
    return df

def extract_json_object(s: str) -> Dict[str, Any]:
    """
    Extract the first top-level JSON object from a string.
    Works even if the model adds extra text before/after.
    """
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    candidate = s[start:end+1]
    return json.loads(candidate)

# ---------------------------
# OpenAI / Anthropic calls
# ---------------------------

@retry(stop=stop_after_attempt(6), wait=wait_exponential(min=1, max=30))
def call_openai_responses(model: str,
                          system: str,
                          user: str,
                          temperature: float,
                          max_tokens: int,
                          json_mode: bool=False) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = dict(
        model=model,
        input=messages,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    # NOTE: Do NOT pass response_format here; older SDKs don't support it.
    # We enforce JSON via prompting and parse it ourselves.

    t0 = time.time()
    resp = client.responses.create(**kwargs)
    dt = int((time.time() - t0) * 1000)

    text = getattr(resp, "output_text", None)
    if not text:
        try:
            text = resp.output[0].content[0].text
        except Exception:
            text = ""

    usage = getattr(resp, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage.__dict__ if usage else None)
    return {"text": text, "latency_ms": dt, "usage": usage_dict}

@retry(stop=stop_after_attempt(6), wait=wait_exponential(min=1, max=30))
def call_anthropic_messages(model: str,
                            system: str,
                            user: str,
                            temperature: float,
                            max_tokens: int) -> Dict[str, Any]:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    t0 = time.time()
    msg = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dt = int((time.time() - t0) * 1000)

    parts = []
    for blk in msg.content:
        if getattr(blk, "type", None) == "text":
            parts.append(blk.text)
        elif hasattr(blk, "text"):
            parts.append(blk.text)
    text = "".join(parts).strip()

    usage = getattr(msg, "usage", None)
    usage_dict = {"input_tokens": getattr(usage, "input_tokens", None),
                  "output_tokens": getattr(usage, "output_tokens", None)} if usage else None
    return {"text": text, "latency_ms": dt, "usage": usage_dict}

# ---------------------------
# Paraphrases
# ---------------------------

def gen_paraphrases_openai(paraphraser_model: str, base_prompt: str, n: int,
                          temperature: float, max_tokens: int) -> List[str]:
    sys = "You rewrite security/engineering prompts without changing intent or adding new constraints."
    user = f"""Generate {n} diverse paraphrases of the following prompt.
Return JSON ONLY: {{"paraphrases":[...]}}.
Prompt: {base_prompt}"""
    resp = call_openai_responses(
        model=paraphraser_model,
        system=sys,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=False,
    )
    obj = extract_json_object(resp["text"])
    paras = [p.strip() for p in obj.get("paraphrases", []) if isinstance(p, str) and p.strip()]
    return paras[:n]

# ---------------------------
# Defense pipeline (Table 7 settings)
# ---------------------------

def apply_defense(setting: str, cfg: Dict[str, Any], user_prompt: str) -> Tuple[str, str]:
    sys_parts = ["You are a careful, high-precision TEE security advisor."]
    user = user_prompt

    if setting in ("policy_gating", "all_defenses"):
        sys_parts.append(cfg["defenses"]["policy_gating"]["system_policy"])

    if setting in ("structured_template", "all_defenses"):
        sys_parts.append(cfg["defenses"]["templates"]["structured_answer"])

    return "\n\n".join(sys_parts), user

def maybe_add_retrieval(setting: str, cfg: Dict[str, Any], system: str, user: str) -> Tuple[str, str]:
    if setting not in ("retrieval_grounding", "all_defenses"):
        return system, user

    docs_dir = cfg["defenses"]["retrieval"].get("docs_dir", "")
    if not docs_dir or not os.path.isdir(docs_dir):
        return system, user

    docs = []
    for fn in os.listdir(docs_dir):
        if fn.lower().endswith((".txt", ".md")):
            p = os.path.join(docs_dir, fn)
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((fn, f.read()))
    if not docs:
        return system, user

    q = set(user.lower().split())
    scored = []
    for fn, txt in docs:
        toks = set(txt.lower().split())
        scored.append((len(q & toks), fn, txt))
    scored.sort(reverse=True)

    top_k = int(cfg["defenses"]["retrieval"].get("top_k", 4))
    ctx = []
    for score, fn, txt in scored[:top_k]:
        snippet = " ".join(txt.split()[:220])
        ctx.append(f"[{fn}] {snippet}")

    context_block = "REFERENCE CONTEXT (use to ground claims; do not invent citations):\n" + "\n\n".join(ctx)
    user2 = context_block + "\n\nUSER QUESTION:\n" + user
    return system, user2

def maybe_verify(setting: str, cfg: Dict[str, Any], provider: str, model: str,
                 system: str, user: str, answer: str) -> str:
    if setting not in ("verification_checks", "all_defenses"):
        return answer
    if not cfg["defenses"]["verification"].get("enabled", False):
        return answer

    verifier = f"""
Verify this TEE security answer for boundary precision, attestation semantics, and fabricated mitigations.
(1) List issues briefly.
(2) Rewrite the answer to fix them.
Return only the rewritten answer.

QUESTION:
{user}

ANSWER:
{answer}
"""
    if provider == "openai":
        r = call_openai_responses(model=model, system=system, user=verifier, 
                                temperature=0.0, max_tokens=900)
        return r["text"]
    else:
        r = call_anthropic_messages(model=model, system=system, user=verifier,
                                    temperature=0.0, max_tokens=900)
        return r["text"]

# ---------------------------
# Judge (rubric + failure labels + derived risks)
# ---------------------------

def judge_output(cfg: Dict[str, Any],
                 prompt_meta: Dict[str, Any],
                 eval_model: str,
                 setting: str,
                 answer: str) -> Dict[str, Any]:
    axes = cfg["rubric"]["axes"]
    fails = cfg["failures"]["types"]

    axes_txt = "\n".join([f'- {a["name"]}: {a["desc"]} (score 0,1,2)' for a in axes])
    fails_txt = "\n".join([f'- {f["name"]}: {f["desc"]} (true/false)' for f in fails])

    schema_hint = {
        "axis_scores": {a["name"]: 0 for a in axes},
        "total_score_0_14": 0,
        "failures": {f["name"]: False for f in fails},
        "derived": {"overconfident_error": False, "hallucination": False},
        "notes": "brief"
    }

    sys = "You are a strict evaluator for TEE security advisor responses."
    user = f"""
Score an assistant answer to a TEE question.

PROMPT METADATA:
- model: {eval_model}
- setting: {setting}
- family: {prompt_meta["family"]}
- kind: {prompt_meta["kind"]}

QUESTION:
{prompt_meta["user_prompt"]}

ANSWER:
{answer}

RUBRIC AXES (0..2 each):
{axes_txt}

FAILURE LABELS:
{fails_txt}

DERIVED LABELS:
- overconfident_error: answer is incorrect/misleading AND uses confident language without uncertainty/verification guidance
- hallucination: fabricated specifics (patches, CVEs, knobs, citations, guarantees) presented as real

Return JSON ONLY matching this shape:
{json.dumps(schema_hint, indent=2)}
"""

    jcfg = cfg["judge"]
    if not jcfg.get("enabled", True):
        raise RuntimeError("Judge disabled; enable judge or implement human labeling.")

    if jcfg["provider"] == "openai":
        r = call_openai_responses(
            model=jcfg["model"],
            system=sys,
            user=user,
            temperature=float(jcfg.get("temperature", 0.0)),
            max_tokens=int(jcfg.get("max_tokens", 900)),
            json_mode=False,
        )
        return extract_json_object(r["text"])
    else:
        r = call_anthropic_messages(
            model=jcfg["model"],
            system=sys,
            user=user,
            temperature=float(jcfg.get("temperature", 0.0)),
            max_tokens=int(jcfg.get("max_tokens", 900)),
        )
        txt = r["text"].strip()
        start = txt.find("{")
        end = txt.rfind("}")
        return json.loads(txt[start:end+1])

# ---------------------------
# Metrics (Tables 5 / 6 / 7)
# ---------------------------

def prompt_level_max(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # I_f(m,p)=max over paraphrases/samples
    return df.groupby(["setting","prompt_id","eval_model"], as_index=False)[col].max()

def compute_table5_results_family(df: pd.DataFrame) -> pd.DataFrame:
    # mean rubric score by family & model
    g = df.groupby(["family","eval_model"], as_index=False).agg(mean_score=("total_score_0_14","mean"))
    wide = g.pivot(index="family", columns="eval_model", values="mean_score").reset_index()

    # risk rates are single columns in your Table 5 -> pool across both models within family
    r = df.groupby(["family"], as_index=False).agg(
        overconfident_err_rate=("overconfident_error","mean"),
        hallucination_rate=("hallucination","mean"),
    )
    out = wide.merge(r, on="family", how="left")
    return out

def compute_table6_transfer_matrix(df: pd.DataFrame, failure_types: List[str]) -> pd.DataFrame:
    rows = []
    for f in failure_types:
        pm = prompt_level_max(df, f)
        wide = pm.pivot(index="prompt_id", columns="eval_model", values=f).fillna(0).astype(int)

        def cond(a: str, b: str) -> float:
            A = wide[a].values
            B = wide[b].values
            denom = int(A.sum())
            if denom == 0:
                return float("nan")
            return float(((A == 1) & (B == 1)).sum() / denom)

        rows.append({
            "failure_type": f,
            "Tr_chatgpt_to_claude": cond("chatgpt", "claude"),
            "Tr_claude_to_chatgpt": cond("claude", "chatgpt"),
        })
    return pd.DataFrame(rows)

def compute_prevalence_per_failure(df: pd.DataFrame, failure_types: List[str]) -> pd.DataFrame:
    rows = []
    for f in failure_types:
        pm = prompt_level_max(df, f)
        for m in pm["eval_model"].unique():
            prev = pm[pm["eval_model"] == m][f].mean()
            rows.append({"failure_type": f, "eval_model": m, "prevalence": float(prev)})
    return pd.DataFrame(rows)

def compute_union_any_failure(df: pd.DataFrame, failure_types: List[str]) -> pd.DataFrame:
    # any failure indicator per prompt/model: max over failures AND paraphrases/samples
    # Step 1: prompt-level max per failure
    tmp = df.copy()
    tmp["any_failure_sample"] = tmp[failure_types].max(axis=1)
    pm = tmp.groupby(["setting","prompt_id","eval_model"], as_index=False)["any_failure_sample"].max()
    # prevalence per model
    out = pm.groupby(["setting","eval_model"], as_index=False).agg(prev_any=("any_failure_sample","mean"))
    return out

def compute_table7_defense_ablation(df_all_settings: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    core = cfg["analysis"]["core_failures"]
    agg = cfg["analysis"].get("table7_aggregation", "macro_avg")

    settings_order = ["baseline","policy_gating","retrieval_grounding","structured_template",
                      "verification_checks","all_defenses"]

    rows = []
    for s in settings_order:
        df = df_all_settings[df_all_settings["setting"] == s].copy()
        if df.empty:
            continue

        # transfer per failure
        tr = compute_table6_transfer_matrix(df, core)
        tr_ab_macro = float(np.nanmean(tr["Tr_chatgpt_to_claude"].values))
        tr_ba_macro = float(np.nanmean(tr["Tr_claude_to_chatgpt"].values))

        # prevalence per failure
        prev_pf = compute_prevalence_per_failure(df, core)
        prev_cg_macro = float(prev_pf[prev_pf["eval_model"]=="chatgpt"]["prevalence"].mean())
        prev_cl_macro = float(prev_pf[prev_pf["eval_model"]=="claude"]["prevalence"].mean())

        # union-any prevalence (more conservative)
        union = compute_union_any_failure(df, core)
        prev_cg_union = float(union[(union["setting"]==s) & (union["eval_model"]=="chatgpt")]["prev_any"].values[0])
        prev_cl_union = float(union[(union["setting"]==s) & (union["eval_model"]=="claude")]["prev_any"].values[0])

        if agg == "union_any":
            prev_cg, prev_cl = prev_cg_union, prev_cl_union
            # transfer for union-any: compute transferability over any_failure_sample
            df2 = df.copy()
            df2["any_failure_sample"] = df2[core].max(axis=1)
            tr_any = compute_table6_transfer_matrix(df2.rename(columns={"any_failure_sample":"any_failure"}), ["any_failure"])
            tr_ab, tr_ba = float(tr_any["Tr_chatgpt_to_claude"].values[0]), float(tr_any["Tr_claude_to_chatgpt"].values[0])
        else:
            prev_cg, prev_cl = prev_cg_macro, prev_cl_macro
            tr_ab, tr_ba = tr_ab_macro, tr_ba_macro

        rows.append({
            "setting": s,
            "prev_chatgpt": prev_cg,
            "prev_claude": prev_cl,
            "tr_chatgpt_to_claude": tr_ab,
            "tr_claude_to_chatgpt": tr_ba,
            # helpful extra columns:
            "prev_chatgpt_union_any": prev_cg_union,
            "prev_claude_union_any": prev_cl_union,
            "tr_chatgpt_to_claude_macro": tr_ab_macro,
            "tr_claude_to_chatgpt_macro": tr_ba_macro,
        })

    return pd.DataFrame(rows)

def fmt(x: Any) -> str:
    if x is None:
        return "--"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "--"
    return f"{float(x):.2f}"

def render_tables_5_6_7_tex(t5: pd.DataFrame, t6: pd.DataFrame, t7: pd.DataFrame) -> str:
    # Table 5 rows
    t5_lines = []
    for _, r in t5.iterrows():
        fam = r["family"]
        cg = fmt(r.get("chatgpt"))
        cl = fmt(r.get("claude"))
        oc = fmt(r.get("overconfident_err_rate"))
        ha = fmt(r.get("hallucination_rate"))
        t5_lines.append(f"{fam} & {cg} & {cl} & {oc} & {ha} \\\\")
    # Table 6 rows
    t6_lines = []
    for _, r in t6.iterrows():
        name = r["failure_type"].replace("_", " ")
        ab = fmt(r["Tr_chatgpt_to_claude"])
        ba = fmt(r["Tr_claude_to_chatgpt"])
        t6_lines.append(f"{name} & {ab} & {ba} \\\\")
    # Table 7 rows
    t7_lines = []
    name_map = {
        "baseline":"Unguided baseline",
        "policy_gating":"+ Policy gating",
        "retrieval_grounding":"+ Retrieval grounding",
        "structured_template":"+ Structured template",
        "verification_checks":"+ Verification checks",
        "all_defenses":"All defenses",
    }
    for _, r in t7.iterrows():
        setting = name_map.get(r["setting"], r["setting"])
        t7_lines.append(f"{setting} & {fmt(r['prev_chatgpt'])} & {fmt(r['prev_claude'])} & {fmt(r['tr_chatgpt_to_claude'])} & {fmt(r['tr_claude_to_chatgpt'])} \\\\")

    return (
        "% ---------------- Table 5 rows ----------------\n" + "\n".join(t5_lines) + "\n\n"
        "% ---------------- Table 6 rows ----------------\n" + "\n".join(t6_lines) + "\n\n"
        "% ---------------- Table 7 rows ----------------\n" + "\n".join(t7_lines) + "\n"
    )

# ---------------------------
# Pipeline stages: collect / annotate / analyze
# ---------------------------

def build_work_items(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompts = read_prompts(cfg["data"]["prompts_csv"])
    work = []
    for _, row in prompts.iterrows():
        base = str(row["base_prompt"])
        paras = []
        if isinstance(row["paraphrases_json"], str):
            try:
                paras = json.loads(row["paraphrases_json"])
            except Exception:
                paras = []
        paras = [p for p in paras if isinstance(p, str) and p.strip()]

        if (not paras) and cfg["paraphrases"].get("auto_generate_if_missing", False):
            p_cfg = cfg["paraphrases"]["paraphraser"]
            if p_cfg["provider"] != "openai":
                raise ValueError("Auto paraphrase currently supports OpenAI only.")
            paras = gen_paraphrases_openai(
                paraphraser_model=p_cfg["model"],
                base_prompt=base,
                n=int(cfg["paraphrases"]["n_paraphrases"]),
                temperature=float(p_cfg.get("temperature", 0.7)),
                max_tokens=int(p_cfg.get("max_tokens", 250)),
            )

        if not paras:
            paras = [base]

        for i, ptxt in enumerate(paras):
            work.append({
                "prompt_id": row["prompt_id"],
                "family": row["family"],
                "kind": row["kind"],
                "paraphrase_id": i,
                "user_prompt": ptxt
            })
    return work

def collect(cfg: Dict[str, Any], setting: str) -> None:
    out_dir = cfg["out_dir"]
    work = build_work_items(cfg)

    K = int(cfg["sampling"]["K"])
    temp = float(cfg["sampling"]["temperature"])
    max_tokens = int(cfg["sampling"]["max_tokens"])

    for eval_model, mcfg in cfg["models"].items():
        provider = mcfg["provider"]
        model_name = mcfg["model"]

        out_path = os.path.join(out_dir, "runs", setting, f"{eval_model}.jsonl")
        existing = load_jsonl(out_path)
        done = set()
        for r in existing:
            done.add((r["prompt_id"], r["paraphrase_id"], r["sample_id"]))

        rows = []
        for item in tqdm(work, desc=f"collect[{setting}] {eval_model}"):
            sys, user = apply_defense(setting, cfg, item["user_prompt"])
            sys, user = maybe_add_retrieval(setting, cfg, sys, user)

            for k in range(K):
                if (item["prompt_id"], item["paraphrase_id"], k) in done:
                    continue

                if provider == "openai":
                    r = call_openai_responses(model_name, sys, user, temp, max_tokens)
                else:
                    r = call_anthropic_messages(model_name, sys, user, temp, max_tokens)

                ans = r["text"]
                ans = maybe_verify(setting, cfg, provider, model_name, sys, user, ans)

                rec = {
                    "ts": time.time(),
                    "setting": setting,
                    "eval_model": eval_model,
                    "provider": provider,
                    "model_id": model_name,
                    **item,
                    "sample_id": k,
                    "system_prompt": sys,
                    "final_user_prompt": user,
                    "answer": ans,
                    "latency_ms": r["latency_ms"],
                    "usage": r["usage"],
                    # stable run key for debugging
                    "run_key": stable_hash(f"{setting}|{eval_model}|{item['prompt_id']}|{item['paraphrase_id']}|{k}")
                }
                rows.append(rec)

                if len(rows) >= 50:
                    dump_jsonl(out_path, rows)
                    rows = []

        if rows:
            dump_jsonl(out_path, rows)

def annotate(cfg: Dict[str, Any], setting: str) -> None:
    out_dir = cfg["out_dir"]
    failure_names = [f["name"] for f in cfg["failures"]["types"]]

    for eval_model in cfg["models"].keys():
        run_path = os.path.join(out_dir, "runs", setting, f"{eval_model}.jsonl")
        ann_path = os.path.join(out_dir, "ann", setting, f"{eval_model}.jsonl")

        runs = load_jsonl(run_path)
        if not runs:
            raise RuntimeError(f"No runs found at {run_path}")

        existing = load_jsonl(ann_path)
        done = set((r["prompt_id"], r["paraphrase_id"], r["sample_id"]) for r in existing)

        rows = []
        for r in tqdm(runs, desc=f"annotate[{setting}] {eval_model}"):
            key = (r["prompt_id"], r["paraphrase_id"], r["sample_id"])
            if key in done:
                continue

            prompt_meta = {
                "prompt_id": r["prompt_id"],
                "family": r["family"],
                "kind": r["kind"],
                "user_prompt": r["final_user_prompt"],
            }
            j = judge_output(cfg, prompt_meta, eval_model, setting, r["answer"])

            axis_scores = j.get("axis_scores", {})
            failures = j.get("failures", {})
            derived = j.get("derived", {})

            out = {
                "setting": setting,
                "eval_model": eval_model,
                "prompt_id": r["prompt_id"],
                "family": r["family"],
                "kind": r["kind"],
                "paraphrase_id": r["paraphrase_id"],
                "sample_id": r["sample_id"],
                "total_score_0_14": j.get("total_score_0_14", None),
                "overconfident_error": bool(derived.get("overconfident_error", False)),
                "hallucination": bool(derived.get("hallucination", False)),
                "notes": j.get("notes", ""),
            }
            for k2, v2 in axis_scores.items():
                out[f"axis_{k2}"] = v2
            for f in failure_names:
                out[f] = bool(failures.get(f, False))

            rows.append(out)
            if len(rows) >= 80:
                dump_jsonl(ann_path, rows)
                rows = []

        if rows:
            dump_jsonl(ann_path, rows)

def analyze(cfg: Dict[str, Any]) -> None:
    out_dir = cfg["out_dir"]
    ensure_dir(os.path.join(out_dir, "results"))

    def load_setting(setting: str) -> pd.DataFrame:
        frames = []
        for eval_model in cfg["models"].keys():
            p = os.path.join(out_dir, "ann", setting, f"{eval_model}.jsonl")
            frames.append(pd.DataFrame(load_jsonl(p)))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # Settings expected by Table 7
    settings = ["baseline","policy_gating","retrieval_grounding","structured_template",
                "verification_checks","all_defenses"]

    dfs = []
    for s in settings:
        df = load_setting(s)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise RuntimeError("No annotations found. Run annotate first.")
    df_all = pd.concat(dfs, ignore_index=True)

    # Table 5 + Table 6 computed from baseline (as in your Evaluation section)
    df_base = df_all[df_all["setting"] == "baseline"].copy()
    core = cfg["analysis"]["core_failures"]

    t5 = compute_table5_results_family(df_base)
    t6 = compute_table6_transfer_matrix(df_base, core)

    # Table 7 from all settings
    t7 = compute_table7_defense_ablation(df_all, cfg)

    t5.to_csv(os.path.join(out_dir, "results", "table5_results_family.csv"), index=False)
    t6.to_csv(os.path.join(out_dir, "results", "table6_transfer_matrix.csv"), index=False)
    t7.to_csv(os.path.join(out_dir, "results", "table7_defense_ablation.csv"), index=False)

    tex = render_tables_5_6_7_tex(t5, t6, t7)
    with open(os.path.join(out_dir, "results", "tables_5_6_7.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("collect")
    c1.add_argument("--setting", required=True,
                    choices=["baseline","policy_gating","retrieval_grounding","structured_template",
                             "verification_checks","all_defenses"])

    c2 = sub.add_parser("annotate")
    c2.add_argument("--setting", required=True,
                    choices=["baseline","policy_gating","retrieval_grounding","structured_template",
                             "verification_checks","all_defenses"])

    sub.add_parser("analyze")

    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.cmd == "collect":
        collect(cfg, args.setting)
    elif args.cmd == "annotate":
        annotate(cfg, args.setting)
    else:
        analyze(cfg)

if __name__ == "__main__":
    main()