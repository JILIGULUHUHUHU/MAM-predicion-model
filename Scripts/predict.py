"""
Predict MAM localization for new protein sequences.
Accepts FASTA file or direct sequence input.
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.esm_classifier import ESM2MAMClassifier
from transformers import AutoTokenizer


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def read_fasta(fasta_path):
    """Read sequences from a FASTA file."""
    sequences = []
    headers = []
    current_header = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append("".join(current_seq))
                    headers.append(current_header)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header is not None:
            sequences.append("".join(current_seq))
            headers.append(current_header)

    return headers, sequences


@torch.no_grad()
def predict_sequences(model, tokenizer, sequences, device, batch_size=8, max_length=1024):
    """Predict MAM localization for a list of sequences."""
    model.eval()
    results = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        results.extend(probs.tolist())

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict MAM localization")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--fasta", default=None, help="Input FASTA file")
    parser.add_argument("--sequence", default=None, help="Single amino acid sequence")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ESM2MAMClassifier(
        model_name=model_cfg["name"],
        hidden_dims=model_cfg["hidden_dims"],
        dropout=model_cfg["dropout"],
        pooling=model_cfg["pooling"],
        max_length=model_cfg["max_length"],
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    # Get input sequences
    headers = None
    if args.fasta:
        headers, sequences = read_fasta(args.fasta)
        print(f"Loaded {len(sequences)} sequences from {args.fasta}")
    elif args.sequence:
        sequences = [args.sequence]
        headers = ["input_sequence"]
    else:
        print("Error: provide --fasta or --sequence")
        sys.exit(1)

    # Predict
    probs = predict_sequences(
        model, tokenizer, sequences, device,
        max_length=model_cfg["max_length"],
    )

    # Build results
    results = []
    for i, (seq, prob) in enumerate(zip(sequences, probs)):
        seq_preview = seq[:50] + "..." if len(seq) > 50 else seq
        header = headers[i] if headers else f"seq_{i+1}"
        results.append({
            "id": header,
            "length": len(seq),
            "mam_probability": round(prob, 4),
            "prediction": "MAM" if prob >= args.threshold else "non-MAM",
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print(f"\nPredictions (threshold={args.threshold}):")
    print(df.to_string(index=False))
    print(f"\nResults saved to {args.output}")

    n_mam = (df["prediction"] == "MAM").sum()
    print(f"Summary: {n_mam}/{len(df)} predicted as MAM-localized")


if __name__ == "__main__":
    main()
