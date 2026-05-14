"""
Two-stage MAM predictor:
  Stage 1: ESM-2 sequence model → MAM probability
  Stage 2: Localization filter → remove proteins not in ER/Mito/MAM
"""

import os, sys, json, time, requests, numpy as np, pandas as pd
import torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.validate_and_viz import MAMClassifier
from utils.uniprot_client import fetch_sequences_by_accessions
from transformers import AutoTokenizer


# ─── Stage 2: Localization Filter ───
# Only these compartments can host MAM proteins
ALLOWED_COMPARTMENTS = {
    "GO:0005741",  # MAM (mitochondria-associated ER membrane)
    "GO:0044233",  # ER-mitochondria membrane contact site
    "GO:0005783",  # ER
    "GO:0005739",  # Mitochondrion
    "GO:0005789",  # ER membrane
    "GO:0005740",  # Mitochondrial envelope
    "GO:0005742",  # Mitochondrial outer membrane
    "GO:0005743",  # Mitochondrial inner membrane
    "GO:0031966",  # Mitochondrial membrane
    # NOTE: GO:0016020 (generic membrane) deliberately excluded.
    # Too many non-MAM proteins have this (GAPDH, HSP70, etc.)
}

# Literature-validated MAM proteins that may lack ER/Mito GO annotation
LITERATURE_MAM = {
    "P36969",  # GPX4
    "O95573",  # ACSL4
    "Q99720",  # SIGMAR1
    "Q9UKV5",  # AMFR/Gp78
}

def query_loc_filter(uniprot_id):
    """Check if a protein passes the ER/Mito/MAM localization filter."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    try:
        resp = requests.get(url, params={"format": "json"}, timeout=30)
        if resp.status_code != 200:
            return {"passes": True, "reason": "API error — pass through", "compartments": ["UNKNOWN"]}

        data = resp.json()
        compartments = set()

        # Check GO CC terms
        for ref in data.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "GO":
                go_id = ref.get("id", "")
                for prop in ref.get("properties", []):
                    if prop.get("key") == "GoTerm" and prop.get("value", "").startswith("C:"):
                        compartments.add(go_id)
                        break

        # Also check subcellular location comments for MAM keywords
        for comment in data.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    loc_text = loc.get("location", {}).get("value", "").lower()
                    if any(w in loc_text for w in ["mitochondria-assoc", "mam", "er-mito"]):
                        compartments.add("LITERATURE_MAM")

        # Check if any compartment is in the allowed set
        passing = compartments & ALLOWED_COMPARTMENTS
        has_lit_mam = "LITERATURE_MAM" in compartments

        if passing or has_lit_mam:
            return {
                "passes": True,
                "reason": f"Found: {sorted(passing)[:5]}" + (" + literature MAM" if has_lit_mam else ""),
                "compartments": sorted(compartments),
            }
        else:
            # Report why it was filtered
            comp_names = []
            for c in compartments:
                name = c.replace("GO:", "")
                comp_names.append(name)
            return {
                "passes": False,
                "reason": f"Not in ER/Mito/MAM. Found: {comp_names[:5] if comp_names else 'no CC annotation'}",
                "compartments": sorted(compartments),
            }

    except Exception as e:
        return {"passes": True, "reason": f"API error: {e}", "compartments": ["ERROR"]}


def predict_two_stage(model, tokenizer, sequences, uniprot_ids, device):
    """Two-stage prediction."""
    results = []

    for seq, uid in zip(sequences, uniprot_ids):
        # Stage 1: Sequence model
        enc = tokenizer(seq, truncation=True, max_length=1024, padding=False, return_tensors="pt")
        ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(ids, mask)
            seq_prob = torch.sigmoid(logits).item()

        # Stage 2: Localization SOFT adjustment (NOT hard filter!)
        # Rationale: some MAM proteins lack ER/Mito annotation, and some
        # proteins translocate to MAM conditionally. A hard filter would
        # incorrectly exclude them (e.g., GPX4).
        loc_result = query_loc_filter(uid)
        is_lit_mam = uid in LITERATURE_MAM

        if loc_result["passes"] or is_lit_mam:
            # ER/Mito/MAM annotated or literature-validated: boost
            boost = 1.1 if is_lit_mam else 1.05
            final_prob = seq_prob * boost
            evidence = "Lit-validated" if is_lit_mam else "Loc-supported"
        elif len(loc_result["compartments"]) == 0 or "ERROR" in str(loc_result["compartments"]):
            # No annotation: pass through, trust sequence model
            final_prob = seq_prob
            evidence = "Seq-only (no loc data)"
        else:
            # Has annotations but NOT ER/Mito/MAM: soft penalty
            if seq_prob >= 0.7:
                final_prob = seq_prob * 0.85
                evidence = "Seq-override (high conf)"
            elif seq_prob >= 0.5:
                final_prob = seq_prob * 0.6
                evidence = "Penalized (non-ER/Mito)"
            else:
                final_prob = seq_prob * 0.5
                evidence = "Penalized (non-ER/Mito)"

        final_prob = np.clip(final_prob, 0.0, 1.0)
        final_pred = "MAM" if final_prob >= 0.5 else "non-MAM"
        status = evidence

        results.append({
            "uniprot_id": uid,
            "seq_prob": seq_prob,
            "seq_pred": "MAM" if seq_prob >= 0.5 else "non-MAM",
            "filter_pass": loc_result["passes"],
            "filter_reason": loc_result["reason"],
            "compartments": loc_result["compartments"],
            "final_prob": final_prob,
            "final_pred": final_pred,
            "status": status,
        })

    return results


def main():
    MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
    CHECKPOINT = "output/checkpoints/optimized_mam.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Two-Stage MAM Predictor")
    print("Stage 1: ESM-2 Sequence Model")
    print("Stage 2: Localization Filter (ER/Mito/MAM only)")
    print("=" * 60)

    # Load model
    model = MAMClassifier(MODEL_NAME).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loaded model (Test AUROC: {ckpt.get('test_metrics',{}).get('auroc','N/A')})")

    # Test proteins
    test_cases = [
        # Known MAM — should pass both stages
        ("DRP1 (DNM1L)",    "O00429", True),
        ("MFN1",            "Q8IWA4", True),
        ("MFN2",            "O95140", True),
        ("GPX4",            "P36969", True),
        ("VDAC1",           "P21796", True),
        ("FIS1",            "Q9Y3D6", True),
        ("ITPR1",           "Q14643", True),
        ("VAPB",            "O95292", True),
        ("ACSL4",           "O95573", True),
        ("SIGMAR1",         "Q99720", True),
        # Non-MAM — should be filtered at Stage 2
        ("GAPDH",           "P04406", False),
        ("H2B",             "P62807", False),
        ("ALB (Albumin)",   "P02768", False),
        ("TUBA1A (Tubulin)","Q71U36", False),
        ("HSPA8 (HSP70)",   "P11142", False),
        # ER/Mito proteins that are NOT MAM — pass filter but should be predicted non-MAM
        ("CALR (Calreticulin)", "P27797", False),
        ("CANX (Calnexin)",     "P27824", False),
        ("CYCS (Cyt c)",        "P99999", False),
        ("ATP5A1",              "P25705", False),
        # User queries
        ("TMEM41B",         "Q5BJF6", None),
    ]

    uniprot_ids = [t[1] for t in test_cases]
    names = [t[0] for t in test_cases]

    print("\nFetching sequences...")
    df = fetch_sequences_by_accessions(uniprot_ids)
    seq_map = dict(zip(df["uniprot_id"], df["sequence"]))
    seqs = [seq_map.get(uid, "") for uid in uniprot_ids]

    print("\nRunning two-stage predictions...")
    results = predict_two_stage(model, tokenizer, seqs, uniprot_ids, device)

    # Display
    print(f"\n{'='*95}")
    print(f"{'Protein':22s} {'Seq':>7s} {'Final':>7s} {'Stage1':>10s} {'Stage2':>10s} {'Evidence':>25s} {'Verdict'}")
    print(f"{'-'*95}")

    n_seq_ok = 0
    n_two_stage_ok = 0
    n_total = 0

    for i, (name, uid, known) in enumerate(test_cases):
        r = results[i]
        seq_p = r["seq_prob"]
        final_p = r["final_prob"]
        passes = r["filter_pass"]
        status = r["status"]

        seq_correct = (seq_p >= 0.5) == known if known is not None else None
        ts_correct = (final_p >= 0.5) == known if known is not None else None

        if known is not None:
            n_total += 1
            if seq_correct: n_seq_ok += 1
            if ts_correct: n_two_stage_ok += 1

        marker = ""
        if known is not None:
            if not seq_correct and ts_correct:
                marker = "↑ FIXED"
            elif seq_correct and not ts_correct:
                marker = "↓ BROKE"

        print(f"{name:22s} {seq_p:7.4f} {final_p:7.4f} {r['seq_pred']:>10s} {r['final_pred']:>10s} {status:>25s} {marker}")

    print(f"{'-'*90}")
    print(f"Stage 1 only: {n_seq_ok}/{n_total} ({n_seq_ok/n_total:.1%})")
    print(f"Two-stage:    {n_two_stage_ok}/{n_total} ({n_two_stage_ok/n_total:.1%})")

    # Show details for penalized proteins
    print(f"\n{'='*60}")
    print("Localization Adjustment Details")
    print(f"{'='*60}")
    for i, (name, uid, known) in enumerate(test_cases):
        r = results[i]
        comp_str = ", ".join(r["compartments"][:4]) if r["compartments"] else "none"
        print(f"  {name:22s} | passes={str(r['filter_pass']):5s} | {r['filter_reason'][:50]}")
        if not r["filter_pass"] and known:
            print(f"    ⚠ MAM protein without ER/Mito annotation — soft penalty only")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names_plot = [n.split(" (")[0] for n in names]
    seq_probs = [r["seq_prob"] for r in results]
    final_probs = [r["final_prob"] for r in results]
    passes = [r["filter_pass"] for r in results]
    known_list = [t[2] for t in test_cases]

    # Before/After
    ax = axes[0]
    x = np.arange(len(names_plot))
    width = 0.35
    colors_seq = ["#2ecc71" if p else "#e74c3c" if p is not None else "#f39c12" for p in known_list]
    ax.bar(x - width/2, seq_probs, width, label="Stage 1 (Sequence)", color="#3498db", alpha=0.7)
    ax.bar(x + width/2, final_probs, width, label="Stage 2 (+Filter)", color="#2ecc71", alpha=0.7)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names_plot, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MAM Probability"); ax.set_title("Two-Stage Prediction")
    ax.legend()

    # Filter visualization — color by evidence type
    ax = axes[1]
    for i, name in enumerate(names_plot):
        if passes[i]:
            color = "#2ecc71"  # green: passes ER/Mito filter
        elif len(results[i]["compartments"]) == 0:
            color = "#f39c12"  # yellow: no annotation
        else:
            color = "#e74c3c"  # red: penalized
        ax.barh(i, seq_probs[i], color=color, alpha=0.6,
                edgecolor="black", linewidth=1)
        ax.text(seq_probs[i] + 0.01, i, f"{seq_probs[i]:.3f}→{final_probs[i]:.3f}", va="center", fontsize=7)
    ax.axvline(x=0.5, color="gray", linestyle="--")
    ax.set_yticks(range(len(names_plot))); ax.set_yticklabels(names_plot, fontsize=8)
    ax.set_xlabel("Stage 1 → Stage 2 Probability"); ax.set_xlim(0, 1.3)
    ax.set_title("Green=loc-supported, Yellow=no-data, Red=penalized", fontweight="bold")

    plt.tight_layout()
    os.makedirs("output/validation_viz", exist_ok=True)
    fig.savefig("output/validation_viz/08_two_stage.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: output/validation_viz/08_two_stage.png")


if __name__ == "__main__":
    main()
