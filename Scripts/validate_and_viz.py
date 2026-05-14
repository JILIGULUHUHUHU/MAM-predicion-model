"""
Validate trained MAM predictor on key proteins: DRP1, MFN1, MFN2, GPX4.
Visualize predictions, attention weights, and embedding space.
"""

import os, sys, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.uniprot_client import fetch_sequences_by_accessions
from transformers import AutoTokenizer, AutoModel


# ─── Attention Pooling ───
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)
    def forward(self, hidden, mask):
        scores = self.query(hidden).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -6e4)
        weights = F.softmax(scores, dim=-1)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1), weights


# ─── Model ───
class MAMClassifier(nn.Module):
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D", hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size
        self.pooler = AttentionPooling(self.embed_dim)
        layers = []
        prev_dim = self.embed_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.GELU(), nn.LayerNorm(h_dim), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Dropout(dropout * 0.5))
        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, return_attention=False, return_embedding=False):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled, attn_weights = self.pooler(hidden, attention_mask)
        logits = self.classifier(pooled)
        outs = (logits,)
        if return_attention: outs += (attn_weights,)
        if return_embedding: outs += (pooled,)
        return outs if len(outs) > 1 else outs[0]


# ─── Target proteins ───
TARGETS = {
    "DRP1 (DNM1L)":  {"id": "O00429", "known_mam": True,  "desc": "Mitochondrial fission GTPase, recruited to MAM"},
    "MFN1":          {"id": "Q8IWA4", "known_mam": True,  "desc": "Mitofusin-1, MAM tether"},
    "MFN2":          {"id": "O95140", "known_mam": True,  "desc": "Mitofusin-2, MAM tether"},
    "GPX4":          {"id": "P36969", "known_mam": True,  "desc": "Glutathione peroxidase 4, MAM-associated ferroptosis regulator"},
    "FIS1":          {"id": "Q9Y3D6", "known_mam": True,  "desc": "Mitochondrial fission 1, MAM fission machinery"},
    "VDAC1":         {"id": "P21796", "known_mam": True,  "desc": "Voltage-dependent anion channel 1, MAM Ca2+ transport"},
    "IP3R1 (ITPR1)": {"id": "Q14643", "known_mam": True,  "desc": "IP3 receptor 1, ER Ca2+ release at MAM"},
    "VAPB":          {"id": "O95292", "known_mam": True,  "desc": "VAPB, ER-Mito tether"},
}

CONTROLS = {
    "GAPDH":     {"id": "P04406", "known_mam": False, "desc": "Glyceraldehyde-3-phosphate dehydrogenase, cytosolic"},
    "H2B":       {"id": "P62807", "known_mam": False, "desc": "Histone H2B, nuclear"},
    "ATP5A1":    {"id": "P25705", "known_mam": False, "desc": "ATP synthase subunit alpha, mitochondrial matrix"},
    "ALB":       {"id": "P02768", "known_mam": False, "desc": "Serum albumin, secreted"},
    "TUBA1A":    {"id": "Q71U36", "known_mam": False, "desc": "Tubulin alpha-1A, cytoskeletal"},
    "HSPA8":     {"id": "P11142", "known_mam": False, "desc": "Heat shock protein 70, cytosolic"},
    "RPL3":      {"id": "P39023", "known_mam": False, "desc": "Ribosomal protein L3, ribosomal"},
    "CYCS":      {"id": "P99999", "known_mam": False, "desc": "Cytochrome c, mitochondrial intermembrane space"},
}


def predict_with_attention(model, tokenizer, sequences, device, max_length=1024):
    """Predict and return attention weights."""
    model.eval()
    results = []
    for seq in sequences:
        enc = tokenizer(seq, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits, attn_weights, embedding = model(ids, mask, return_attention=True, return_embedding=True)
            prob = torch.sigmoid(logits).item()
            attn = attn_weights[0].cpu().numpy()  # (L,)
            emb = embedding[0].cpu().numpy()

        # Remove special tokens (CLS=0, EOS=2)
        token_ids = ids[0].cpu().numpy()
        valid_mask = (token_ids != 0) & (token_ids != 2)
        aa_attn = attn[valid_mask]
        seq_length = valid_mask.sum()

        results.append({
            "probability": prob,
            "attention": aa_attn,
            "sequence": seq[:seq_length],
            "embedding": emb,
            "length": seq_length,
        })
    return results


def plot_predictions(results_dict, output_path):
    """Bar chart of MAM probabilities for all proteins."""
    names = list(results_dict.keys())
    probs = [results_dict[n]["probability"] for n in names]
    is_mam = [results_dict[n]["known_mam"] for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2ecc71" if m else "#e74c3c" for m in is_mam]
    bars = ax.barh(names, probs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Decision threshold (0.5)")

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{prob:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("MAM Probability", fontsize=12)
    ax.set_title("MAM Localization Prediction — Key Proteins", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.15)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ecc71", label="Known MAM"), Patch(facecolor="#e74c3c", label="Control (non-MAM)")]
    ax.legend(handles=legend_elements + [ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)], loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_attention_maps(results_dict, output_path):
    """Attention weight heatmaps for each protein (residue-level)."""
    mam_proteins = {k: v for k, v in results_dict.items() if v["known_mam"]}

    n = len(mam_proteins)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.5 * n))
    if n == 1: axes = [axes]

    for ax, (name, data) in zip(axes, mam_proteins.items()):
        attn = data["attention"]
        seq_len = len(attn)

        # Normalize attention to [0, 1] for display
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-9)

        # Create a colored bar for each residue
        cmap = plt.cm.YlOrRd
        colors = cmap(attn_norm)

        x = np.arange(seq_len)
        ax.bar(x, attn_norm, width=1.0, color=colors, edgecolor="none")

        # Mark top 5% attention positions
        threshold = np.percentile(attn_norm, 95)
        top_positions = np.where(attn_norm >= threshold)[0]
        for pos in top_positions:
            ax.axvline(x=pos, color="blue", alpha=0.3, linewidth=0.3)

        ax.set_title(f"{name} ({seq_len} aa) — Residue Attention Weights", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residue Position")
        ax.set_ylabel("Attention")
        ax.set_xlim(0, seq_len)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_attention_zoom(results_dict, output_path, window=50):
    """Zoomed attention for N-terminal regions of DRP1, MFN1, MFN2, GPX4."""
    key_proteins = ["DRP1 (DNM1L)", "MFN1", "MFN2", "GPX4"]
    available = [k for k in key_proteins if k in results_dict]

    fig, axes = plt.subplots(len(available), 1, figsize=(16, 2.5 * len(available)))
    if len(available) == 1: axes = [axes]

    for ax, name in zip(axes, available):
        data = results_dict[name]
        seq = data["sequence"]
        attn = data["attention"]

        # Show N-terminal region
        end = min(window, len(seq))
        x = np.arange(end)
        ax.bar(x, attn[:end], width=0.8, color="#3498db", edgecolor="none")

        # Add amino acid labels for high-attention positions
        threshold = np.percentile(attn[:end], 80)
        for i in range(end):
            if attn[i] >= threshold:
                ax.annotate(seq[i], (i, attn[i]), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=7, rotation=90, color="red")

        ax.set_title(f"{name} — N-terminal Attention (first {window} residues)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residue Position"); ax.set_ylabel("Attention")
        ax.set_xlim(0, end)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_embedding_tsne(results_dict, output_path):
    """t-SNE visualization of protein embeddings."""
    from sklearn.manifold import TSNE

    names = list(results_dict.keys())
    embeddings = np.stack([results_dict[n]["embedding"] for n in names])
    is_mam = [results_dict[n]["known_mam"] for n in names]
    probs = [results_dict[n]["probability"] for n in names]

    tsne = TSNE(n_components=2, perplexity=min(5, len(names)-1), random_state=42, max_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#2ecc71" if m else "#e74c3c" for m in is_mam]
    sizes = [max(80, p * 400) for p in probs]

    for i, name in enumerate(names):
        ax.scatter(emb_2d[i, 0], emb_2d[i, 1], c=colors[i], s=sizes[i],
                  edgecolors="black", linewidth=1, alpha=0.8, zorder=3)
        ax.annotate(name.split(" (")[0], (emb_2d[i, 0], emb_2d[i, 1]),
                   textcoords="offset points", xytext=(8, 4), fontsize=10, fontweight="bold")

    ax.set_xlabel("t-SNE 1", fontsize=12); ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Protein Embedding Space (t-SNE) — Size ∝ MAM Probability", fontsize=14, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2ecc71", label="Known MAM"), Patch(facecolor="#e74c3c", label="Control")])

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_attention_logo(results_dict, output_path, top_k=20):
    """Sequence logo-style visualization of high-attention motifs."""
    key_proteins = ["DRP1 (DNM1L)", "MFN1", "MFN2", "GPX4"]
    available = [k for k in key_proteins if k in results_dict]

    fig, axes = plt.subplots(len(available), 1, figsize=(16, 3 * len(available)))
    if len(available) == 1: axes = [axes]

    aa_colors = {
        'A': '#8dd3c7', 'C': '#ffffb3', 'D': '#bebada', 'E': '#fb8072',
        'F': '#80b1d3', 'G': '#fdb462', 'H': '#b3de69', 'I': '#fccde5',
        'K': '#d9d9d9', 'L': '#bc80bd', 'M': '#ccebc5', 'N': '#ffed6f',
        'P': '#b3cde3', 'Q': '#ccebc5', 'R': '#d9d9d9', 'S': '#fbb4ae',
        'T': '#b3cde3', 'V': '#ccebc5', 'W': '#80b1d3', 'Y': '#fdb462',
    }

    for ax, name in zip(axes, available):
        data = results_dict[name]
        seq = data["sequence"]
        attn = data["attention"]

        # Find top-k attention windows
        top_idx = np.argsort(attn)[-top_k:]
        top_idx.sort()

        for i, idx in enumerate(top_idx):
            aa = seq[idx] if idx < len(seq) else "X"
            color = aa_colors.get(aa, "#cccccc")
            ax.bar(i, attn[idx], color=color, edgecolor="gray", linewidth=0.3)
            ax.annotate(aa, (i, attn[idx]), textcoords="offset points",
                       xytext=(0, 5), ha="center", fontsize=10, fontweight="bold", color="black")

        ax.set_title(f"{name} — Top {top_k} Attention Positions", fontsize=11, fontweight="bold")
        ax.set_xlabel("Rank"); ax.set_ylabel("Attention")
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([str(idx+1) for idx in top_idx], rotation=90, fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
    CHECKPOINT = "output/checkpoints/optimized_mam.pt"
    VIZ_DIR = "output/validation_viz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model
    print(f"Loading model from {CHECKPOINT}...")
    model = MAMClassifier(MODEL_NAME).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Test AUROC (from saved): {ckpt.get('test_metrics', {}).get('auroc', 'N/A')}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fetch sequences
    all_targets = {**TARGETS, **CONTROLS}
    accessions = [v["id"] for v in all_targets.values()]
    print(f"\nFetching {len(accessions)} protein sequences...")
    df = fetch_sequences_by_accessions(accessions)
    seq_map = dict(zip(df["uniprot_id"], df["sequence"]))
    print(f"  Got {len(seq_map)} sequences")

    # Build results dict
    results_dict = {}
    seqs_to_predict = []
    names_order = []
    for name, info in all_targets.items():
        acc = info["id"]
        if acc in seq_map:
            seq = seq_map[acc]
            results_dict[name] = {"known_mam": info["known_mam"], "sequence": seq, "desc": info["desc"]}
            seqs_to_predict.append(seq)
            names_order.append(name)
        else:
            print(f"  WARNING: {name} ({acc}) not found")

    # Predict
    print(f"\nRunning predictions with attention...")
    preds = predict_with_attention(model, tokenizer, seqs_to_predict, device)

    for name, pred in zip(names_order, preds):
        results_dict[name].update(pred)
        status = "MAM ✓" if pred["probability"] >= 0.5 else "non-MAM"
        expected = "✓" if results_dict[name]["known_mam"] else "  "
        marker = "✓" if (pred["probability"] >= 0.5) == results_dict[name]["known_mam"] else "✗ MISMATCH"
        print(f"  {marker} {name:20s} | prob={pred['probability']:.4f} | {status:8s} | expected={expected} | {results_dict[name]['desc'][:50]}")

    # Create output dir
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Generate all visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    plot_predictions(results_dict, os.path.join(VIZ_DIR, "01_predictions.png"))
    plot_attention_maps(results_dict, os.path.join(VIZ_DIR, "02_attention_maps.png"))
    plot_attention_zoom(results_dict, os.path.join(VIZ_DIR, "03_attention_zoom.png"))
    plot_attention_logo(results_dict, os.path.join(VIZ_DIR, "04_attention_logo.png"))
    plot_embedding_tsne(results_dict, os.path.join(VIZ_DIR, "05_embedding_tsne.png"))

    # Save numeric results
    summary = {}
    for name, data in results_dict.items():
        summary[name] = {"probability": round(float(data["probability"]), 4), "known_mam": bool(data["known_mam"]), "correct": bool((data["probability"] >= 0.5) == data["known_mam"]), "length": int(data["length"])}
    n_correct = sum(1 for v in summary.values() if v["correct"])
    n_total = len(summary)
    summary["accuracy"] = round(n_correct / n_total, 4)
    json.dump(summary, open(os.path.join(VIZ_DIR, "results.json"), "w"), indent=2)
    print(f"\nResults saved to {VIZ_DIR}/")
    print(f"\nAccuracy: {n_correct}/{n_total} ({n_correct/n_total:.1%})")
    print(f"All visualizations: {VIZ_DIR}/")

if __name__ == "__main__":
    main()
