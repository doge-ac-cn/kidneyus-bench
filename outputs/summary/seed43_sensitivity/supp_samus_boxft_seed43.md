# Supplementary arm — SAMUS box-prompt fine-tuning (registered seed 43)

Same recipe as the reported light fine-tune; prompts = GT boxes, prompt encoder unfrozen (43.72M); AdamW 1e-4 cosine, 50 ep, bs 8.

| eval | prompt | box-FT | point-trained (seed43) | delta |
|---|---|---|---|---|
| E1 | p3 | **0.9464** | 0.9126 | +0.0338 |
| E1 | p4 | **0.9188** | 0.9002 | +0.0186 |
| E1 | p2 | **0.8507** | 0.8771 | -0.0264 |
| E2 | p3 | **0.892** | 0.7902 | +0.1018 |
| E2 | p4 | **0.4391** | 0.4258 | +0.0133 |
| E2 | p2 | **0.4602** | 0.4713 | -0.0111 |

Box-pathway fine-tuning closes the asymmetry: external oracle-box Dice rises to 0.892 (level with MedSAM 0.884); the detection-box gap remains set by the boxes themselves.
