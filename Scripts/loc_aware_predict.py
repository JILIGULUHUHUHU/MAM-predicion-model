"""
Localization-aware MAM predictor.
Combines ESM-2 sequence model with UniProt subcellular localization annotations.
Post-hoc adjustment — no retraining needed.
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


# ─── Localization → MAM prior mapping ───
# GO CC terms and their relevance to MAM localization
LOC_PRIORS = {
    # GO term: (boost_factor, description)
    "GO:0005741": (1.5, "MAM (direct evidence)"),        # MAM itself
    "GO:0044233": (1.3, "ER-mito MCS"),                   # ER-mito contact site
    "GO:0005739": (1.1, "Mitochondrion"),                 # mitochondrial
    "GO:0005783": (1.1, "ER"),                            # ER
    "GO:0005740": (1.1, "Mitochondrial envelope"),
    "GO:0005743": (1.1, "Mitochondrial inner membrane"),
    "GO:0005742": (1.0, "Mitochondrial outer membrane (MAM-proximal)"),
    "GO:0005789": (1.05, "ER membrane"),
    "GO:0005829": (0.7, "Cytosol (penalize)"),            # unlikely to be MAM
    "GO:0005634": (0.5, "Nucleus (penalize)"),            # strongly penalize
    "GO:0005576": (0.4, "Extracellular (penalize)"),      # strongly penalize
}

# Subcellular location keywords from UniProt
LOC_KEYWORDS = {
    "MAM": 1.5,
    "Mitochondrion": 1.1,
    "Endoplasmic reticulum": 1.1,
    "Mitochondrion outer membrane": 1.05,
    "Cytoplasm": 0.8,
    "Nucleus": 0.5,
    "Secreted": 0.3,
    "Extracellular": 0.4,
}


def query_uniprot_localization(uniprot_id):
    """Query UniProt for subcellular location annotations of a protein."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    try:
        resp = requests.get(url, params={"format": "json"}, timeout=30)
        if resp.status_code != 200:
            return {"go_terms": [], "keywords": [], "comments": []}

        data = resp.json()

        # Extract GO CC terms
        go_terms = []
        for ref in data.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "GO":
                for prop in ref.get("properties", []):
                    if prop.get("key") == "GoTerm" and prop.get("value", "").startswith("C:"):
                        go_terms.append(ref["id"])  # e.g. "GO:0005739"

        # Extract subcellular location keywords
        keywords = []
        for kw in data.get("keywords", []):
            keywords.append(kw.get("name", ""))

        # Extract subcellular location comments
        comments = []
        for comment in data.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    comments.append(loc.get("location", {}).get("value", ""))

        return {"go_terms": go_terms, "keywords": keywords, "comments": comments}

    except Exception as e:
        print(f"    UniProt query error: {e}")
        return {"go_terms": [], "keywords": [], "comments": []}


def compute_localization_factor(loc_data):
    """
    Compute localization adjustment factor.
    > 1.0 = boost MAM probability
    < 1.0 = reduce MAM probability
    1.0 = no adjustment
    """
    factors = []

    # Check GO terms
    for go_term in loc_data.get("go_terms", []):
        if go_term in LOC_PRIORS:
            factors.append(LOC_PRIORS[go_term][0])

    # Check keywords
    for kw in loc_data.get("keywords", []):
        if kw in LOC_KEYWORDS:
            factors.append(LOC_KEYWORDS[kw])

    # Check comments for subcellular location text
    for comment in loc_data.get("comments", []):
        comment_lower = comment.lower()
        if "mitochondria-associated" in comment_lower or "mam" in comment_lower:
            factors.append(1.5)
        elif "mitochondrion" in comment_lower:
            factors.append(1.1)
        elif "endoplasmic reticulum" in comment_lower:
            factors.append(1.1)
        elif "nucleus" in comment_lower:
            factors.append(0.5)
        elif "secreted" in comment_lower:
            factors.append(0.3)

    if not factors:
        return 1.0, "No localization annotation — sequence-only prediction"

    # Key insight: a protein annotated to BOTH ER and mitochondrion is
    # much more likely to be MAM than either alone
    has_er = any("ER" in str(LOC_PRIORS.get(g, ("", ""))[1]) for g in loc_data.get("go_terms", [])) or \
             "Endoplasmic reticulum" in str(loc_data.get("keywords", []))
    has_mito = any("Mitochondrion" in str(LOC_PRIORS.get(g, ("", ""))[1]) for g in loc_data.get("go_terms", [])) or \
               "Mitochondrion" in str(loc_data.get("keywords", []))

    if has_er and has_mito:
        factors.append(1.2)  # ER+mito co-localization bonus

    # Combine factors: geometric mean, capped
    factor = np.exp(np.mean(np.log([max(0.1, f) for f in factors])))
    factor = np.clip(factor, 0.2, 1.8)

    # Describe evidence
    boost_desc = []
    for go_term in loc_data.get("go_terms", [])[:5]:
        if go_term in LOC_PRIORS:
            boost_desc.append(LOC_PRIORS[go_term][1])
    evidence_str = ", ".join(boost_desc[:4]) if boost_desc else "keyword-based"

    return factor, evidence_str


def predict_with_localization(model, tokenizer, sequences, uniprot_ids, device):
    """Predict MAM with localization-aware adjustment."""
    results = []

    for seq, uid in zip(sequences, uniprot_ids):
        # Sequence prediction
        enc = tokenizer(seq, truncation=True, max_length=1024, padding=False, return_tensors="pt")
        ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(ids, mask)
            seq_prob = torch.sigmoid(logits).item()

        # Localization adjustment
        print(f"  Querying UniProt localization for {uid}...")
        loc_data = query_uniprot_localization(uid)
        loc_factor, loc_evidence = compute_localization_factor(loc_data)

        # Final score: logit-space adjustment (mathematically sound)
        # logit(adj) = logit(seq) + log(loc_factor)
        eps = 1e-8
        logit = np.log(seq_prob + eps) - np.log(1 - seq_prob + eps)
        logit_adj = logit + np.log(max(loc_factor, 0.1))
        adjusted_prob = 1.0 / (1.0 + np.exp(-logit_adj))

        results.append({
            "uniprot_id": uid,
            "seq_prob": seq_prob,
            "loc_factor": loc_factor,
            "loc_evidence": loc_evidence,
            "adjusted_prob": adjusted_prob,
            "go_terms": loc_data["go_terms"][:8],
            "keywords": loc_data["keywords"][:5],
        })

    return results


def main():
    MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
    CHECKPOINT = "output/checkpoints/optimized_mam.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VIZ_DIR = "output/validation_viz"

    print("=" * 60)
    print("Localization-Aware MAM Predictor")
    print("=" * 60)

    # Load model
    model = MAMClassifier(MODEL_NAME).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Test proteins
    test_cases = [
        # Known MAM — should be boosted
        ("DRP1 (DNM1L)",  "O00429", True),
        ("MFN1",          "Q8IWA4", True),
        ("MFN2",          "O95140", True),
        ("GPX4",          "P36969", True),
        ("VDAC1",         "P21796", True),
        ("FIS1",          "Q9Y3D6", True),
        ("ITPR1",         "Q14643", True),
        ("VAPB",          "O95292", True),
        # Non-MAM — should be penalized
        ("GAPDH",         "P04406", False),
        ("H2B (HIST1H2B)","P62807", False),
        ("ALB (Albumin)", "P02768", False),
        # User's query
        ("TMEM41B",       "Q5BJF6", None),  # unknown
        # Additional interesting cases
        ("CYCS (Cyt c)",  "P99999", False),
        ("CALR (Calreticulin)", "P27797", False),  # ER chaperone
        ("HSPA5 (BiP)",   "P11021", False),  # ER chaperone
        ("CANX (Calnexin)","P27824", False),  # ER chaperone
    ]

    uniprot_ids = [t[1] for t in test_cases]
    names = [t[0] for t in test_cases]

    print("\nFetching sequences...")
    df = fetch_sequences_by_accessions(uniprot_ids)
    seq_map = dict(zip(df["uniprot_id"], df["sequence"]))

    seqs = [seq_map.get(uid, "") for uid in uniprot_ids]

    print("\nRunning localization-aware predictions...")
    results = predict_with_localization(model, tokenizer, seqs, uniprot_ids, device)

    # Display
    print(f"\n{'='*80}")
    print(f"{'Protein':22s} {'Seq':>7s} {'LocAdj':>7s} {'Final':>7s} {'SeqPred':>10s} {'FinalPred':>10s} {'Evidence'}")
    print(f"{'-'*80}")

    n_seq_correct = 0
    n_loc_correct = 0
    n_eval = 0

    for i, (name, uid, known) in enumerate(test_cases):
        r = results[i]
        seq_p = r["seq_prob"]
        adj_p = r["adjusted_prob"]
        factor = r["loc_factor"]
        evidence = r["loc_evidence"]

        seq_pred = "MAM" if seq_p >= 0.5 else "non-MAM"
        final_pred = "MAM" if adj_p >= 0.5 else "non-MAM"

        if known is not None:
            n_eval += 1
            if (seq_p >= 0.5) == known: n_seq_correct += 1
            if (adj_p >= 0.5) == known: n_loc_correct += 1

            status = ""
            if (seq_p >= 0.5) != known and (adj_p >= 0.5) == known:
                status = "↑ FIXED by loc"
            elif (seq_p >= 0.5) == known and (adj_p >= 0.5) != known:
                status = "↓ BROKE"
        else:
            status = "(unknown)"

        print(f"{name:22s} {seq_p:7.4f} {factor:7.3f} {adj_p:7.4f} {seq_pred:>10s} {final_pred:>10s} {evidence[:60]} {status}")

    print(f"{'-'*80}")
    if n_eval > 0:
        print(f"Sequence-only: {n_seq_correct}/{n_eval} ({n_seq_correct/n_eval:.1%})")
        print(f"+Localization: {n_loc_correct}/{n_eval} ({n_loc_correct/n_eval:.1%})")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names_plot = [n.split(" (")[0] for n in names]
    seq_probs = [r["seq_prob"] for r in results]
    adj_probs = [r["adjusted_prob"] for r in results]
    known_mam = [t[2] for t in test_cases]

    # Before/After comparison
    ax = axes[0]
    x = np.arange(len(names_plot))
    width = 0.35
    ax.bar(x - width/2, seq_probs, width, label="Sequence-only", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, adj_probs, width, label="+Localization", color="#2ecc71", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names_plot, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAM Probability"); ax.set_title("Effect of Localization Adjustment")
    ax.legend()

    # Scatter: sequence vs adjusted
    ax = axes[1]
    for i in range(len(names_plot)):
        color = "#2ecc71" if known_mam[i] else "#e74c3c" if known_mam[i] is not None else "#f39c12"
        ax.scatter(seq_probs[i], adj_probs[i], c=color, s=100, edgecolors="black", linewidth=1)
        ax.annotate(names_plot[i], (seq_probs[i], adj_probs[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Sequence-only Probability"); ax.set_ylabel("Localization-adjusted Probability")
    ax.set_title("Before vs After Localization Adjustment")

    plt.tight_layout()
    os.makedirs(VIZ_DIR, exist_ok=True)
    fig.savefig(os.path.join(VIZ_DIR, "07_localization_aware.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {VIZ_DIR}/07_localization_aware.png")

    # TMEM41B detail
    tmem_idx = [t[1] for t in test_cases].index("Q5BJF6")
    r_tmem = results[tmem_idx]
    print(f"\n{'='*60}")
    print(f"TMEM41B Detailed Report")
    print(f"{'='*60}")
    print(f"  Sequence probability: {r_tmem['seq_prob']:.4f}")
    print(f"  Localization factor:  {r_tmem['loc_factor']:.3f}")
    print(f"  Adjusted probability: {r_tmem['adjusted_prob']:.4f}")
    print(f"  GO terms: {r_tmem['go_terms']}")
    print(f"  Keywords: {r_tmem['keywords']}")
    print(f"  Evidence: {r_tmem['loc_evidence']}")
    print(f"  Final prediction: {'MAM' if r_tmem['adjusted_prob'] >= 0.5 else 'non-MAM'}")

    # Proposal for model integration
    print(f"""
{'='*60}
How to Integrate Localization into Training (Future)
{'='*60}

For retraining the model with localization features:

1. ADD LOCALIZATION FEATURES TO DATASET:
   For each protein, create a 10-dim binary vector encoding:
   [MAM, ER, Mito, ER_membrane, Mito_OM, Cytosol, Nucleus,
    PM, Extracellular, Other]

2. ARCHITECTURE:
   ESM-2 → [640-dim embedding]
   Localization → [10-dim one-hot]
   Concatenate → [650-dim] → Classifier Head

3. HANDLING MISSING ANNOTATIONS:
   - During training: use known annotations
   - During inference: use UniProt API to fetch annotations
   - For proteins without annotations: use all-zero vector
     (model learns to rely more on sequence when localization
      features are absent)

4. EXPECTED IMPROVEMENT:
   - Eliminates false positives from non-ER/mito compartments
   - Boosts true MAM proteins annotated to both ER and mito
   - Particularly helps with ER-only proteins that are NOT MAM
""")


if __name__ == "__main__":
    main()
