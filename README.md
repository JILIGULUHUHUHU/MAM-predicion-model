# MAM Protein Localization Predictor

Deep learning model for predicting whether a protein localizes to Mitochondria-Associated ER Membrane (MAM).

## Model Architecture

```
ESM-2 t30 (150M params, 30-layer Transformer)
    → Attention Pooling (learned residue importance)
    → MLP Classifier [512, 256] + Dropout 0.3
    → MAM Probability (0-1)
```

## Available Models

| Model | File | Test AUROC | Architecture |
|-------|------|-----------|--------------|
| **optimized_mam.pt** (recommended) | 595MB | **0.848** | t30_150M + AttnPool + 1814 samples |
| final_mam_v2.pt | 595MB | 0.842 | Same arch, longer training |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Predict a single protein
python Scripts/predict.py \
  --checkpoint Models/optimized_mam.pt \
  --sequence "MAVPPTYADLGKSARDVFTKGYGFGLIKLDLK..."

# Predict from FASTA file
python Scripts/predict.py \
  --checkpoint Models/optimized_mam.pt \
  --fasta your_proteins.fasta \
  --output results.csv
```

## Two-Stage Prediction (Recommended for Production)

```bash
python Scripts/two_stage_predict.py
```

Pipeline:
1. **Stage 1**: ESM-2 sequence model → MAM probability
2. **Stage 2**: Subcellular localization soft adjustment
   - ER/Mito/MAM annotated: boost
   - Non-ER/Mito annotated: penalty (sequence model can override if confident)
   - No annotation: trust sequence model

The two-stage approach eliminates false positives from non-ER/Mito compartments
(TUBA1A, HSPA8, GAPDH) while allowing sequence-confident predictions through.

## Validation Results (16 proteins)

### Key MAM Proteins
| Protein | Probability | Prediction |
|---------|------------|------------|
| MFN1 | 0.869 | MAM |
| MFN2 | 0.874 | MAM |
| DRP1 (DNM1L) | 0.683 | MAM |
| VDAC1 | 0.997 | MAM |
| FIS1 | 0.925 | MAM |
| ITPR1 | 0.608 | MAM |
| VAPB | 0.561 | MAM |

### GPX4 — Special Case
GPX4 localizes to MAM via protein-protein interactions (not sequence signals).
Sequence model gives low probability (0.247). This protein requires the literature
evidence tier for correct classification. See `Scripts/hybrid_predict.py`.

### Non-MAM Controls
| Protein | Seq Prob | After Filter | Fixed? |
|---------|---------|-------------|--------|
| GAPDH | 0.392 | 0.196 non-MAM | Yes |
| H2B | 0.211 | 0.105 non-MAM | Yes |
| TUBA1A | 0.654 MAM | 0.392 non-MAM | Yes ↑ |
| HSPA8 | 0.558 MAM | 0.335 non-MAM | Yes ↑ |
| ALB | 0.614 MAM | 0.645 MAM | No (ER-annotated) |

## Visualizations

| File | Description |
|------|-------------|
| 01_predictions.png | Bar chart of MAM probabilities |
| 02_attention_maps.png | Residue-level attention weights for MAM proteins |
| 03_attention_zoom.png | N-terminal attention zoom (DRP1, MFN1, MFN2, GPX4) |
| 04_attention_logo.png | Top attention positions with amino acid labels |
| 05_embedding_tsne.png | t-SNE of protein embeddings |
| 06_hybrid_model.png | Sequence + PPI + Literature evidence |
| 07_localization_aware.png | Effect of localization adjustment |
| 08_two_stage.png | Two-stage prediction results |

## Dataset

- **1814 sequences**: 454 positive / 1360 negative (1:3 ratio)
- **Positive**: Human + Mouse MAM proteins (GO:0005741 + literature curated)
- **Negative**: Stratified by compartment (mito, ER, cytosol, nucleus, plasma membrane)
- **Clustering**: 3-mer frequency at 30% similarity to prevent homology leakage
- **Split**: Train 1470 / Val 175 / Test 169

## Evidence Tiers

| Tier | Method | Example Proteins |
|------|--------|-----------------|
| Tier 1 | Sequence model | MFN1, MFN2, VDAC1, DRP1, FIS1 |
| Tier 2 | PPI network support | AMFR, borderline cases |
| Tier 3 | Literature evidence | GPX4, ACSL4, SIGMAR1 |

## Limitations

1. **GPX4-like proteins**: Interaction-dependent MAM proteins require literature evidence
2. **ALB (serum albumin)**: False positive due to ER synthesis annotation
3. **Small dataset**: 454 positive samples limits generalization
4. **Model size**: t30_150M requires 6GB+ GPU VRAM

## File Structure

```
MAM-Delivery/
├── README.md
├── requirements.txt
├── Models/
│   ├── optimized_mam.pt    (best: AUROC 0.848)
│   └── final_mam_v2.pt     (backup: AUROC 0.842)
├── Scripts/
│   ├── two_stage_predict.py   (two-stage: seq + localization)
│   ├── predict.py             (simple single-protein prediction)
│   ├── validate_and_viz.py    (validation + attention visualization)
│   ├── hybrid_predict.py      (multi-evidence prediction)
│   ├── loc_aware_predict.py   (localization-aware prediction)
│   └── uniprot_client.py      (UniProt API wrapper)
├── Results/
│   ├── optimized_metrics.json
│   ├── final_v2_metrics.json
│   ├── results.json
│   └── Plots/
│       ├── 01_predictions.png
│       ├── 02_attention_maps.png
│       ├── ...
│       └── 08_two_stage.png
└── Data/ (optional: place expanded_train.csv etc. here)
```
