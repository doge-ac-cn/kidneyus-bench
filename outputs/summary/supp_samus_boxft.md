# Supplementary arm — SAMUS box-prompt fine-tuning (seed 42 point check)

Purpose: remove the prompt-pathway asymmetry declared in DATA_QUALITY §8 (the reported arm was fine-tuned with point prompts only, so its box-pathway rows were zero-adaptation).

**Recipe** (identical to the reported light fine-tune except the prompt and the unfrozen prompt encoder):

- trainable: official SAMUS light modules (cnn_embed, post_pos_embed, Adapter, rel_pos 2/5/8/11, upneck) + mask decoder + **prompt encoder** = 43.72M
- prompts: GT-derived box per training image (same box source as MedSAM training)
- loss: Dice + BCEWithLogits(pos_weight=2); AdamW 1e-4, cosine, 50 epochs, bs 8
- GPU: training 25 min; eval 42 min (V100-16GB)

Internal validation (box path): **0.9534**.

| eval set | prompt | box-FT | point-trained arm (seed42) | delta |
|---|---|---|---|---|
| E1 OKD test | p3 oracle box | **0.9443** | 0.8938 | +0.0505 |
| E1 OKD test | p4 detection box | **0.9165** | 0.8835 | +0.0330 |
| E1 OKD test | p2 point | 0.8262 | 0.8794 | -0.0532 |
| E2 TRUSTED | p3 oracle box | **0.8866** | 0.7081 | **+0.1785** |
| E2 TRUSTED | p4 detection box | **0.4393** | 0.4020 | +0.0373 |
| E2 TRUSTED | p2 point | 0.4616 | 0.4777 | -0.0161 |

**Reading**:

- Fine-tuning the box pathway removes the asymmetry: SAMUS external oracle-box Dice rises from 0.708 to 0.887 (+0.178), now on par with MedSAM (0.884) and SAM+LoRA (0.872) under the same oracle-box protocol.
- The detection-box path (P4) improves only +0.037 (0.402 -> 0.439): with a well-adapted box encoder the remaining gap is set by the detection boxes themselves, consistent with the coverage attribution.
- The point path degrades slightly (E1 -0.053, E2 -0.016) - the expected prompt specialisation trade-off.

Not part of the registered main table; reported as a supplementary arm with the registered inference pathway (`m3_common.infer_samus_box`; empty prompt -> 0).

Provenance: `scripts/supp_samus_boxft.py` (train+eval) -> `outputs/checkpoints/samus_boxft_seed42.pt` (gitignored); `outputs/summary/supp_samus_boxft.json`. Note: an eval-side list-aliasing bug (`p3 = p4 = pt = []`) was found and fixed before these numbers; the fixed values were re-verified by an independent 5-slice spot check.
