# Supplementary item 9 — SAMUS recipe registration check

## Retrieval status

All OSF endpoints returned **HTTP 503** on 2026-10-17 (site outage / WAF):
- `https://osf.io/tx7rn/`
- `https://api.osf.io/v2/registrations/tx7rn/`
- `https://api.osf.io/v2/registrations/tx7rn/files/osfstorage/`
- `https://osf.io/tx7rn/files/`
- `https://osf.io/tx7rn/download`
- `https://r.jina.ai/https://osf.io/tx7rn/`

`REGISTRATION_DOCUMENT_EN.md` could not be retrieved at this time.

## Offline registration basis (frozen document)

plan_A v1.4 (the integrated registration basis) specifies the SAMUS arm as:
> 官方权重直接推理 + OKD轻微微调两版 (official weights direct inference + OKD light fine-tuning, two versions)

No layer / learning-rate / epoch configuration is stated in this basis.

## Reported arm

- Official SAMUS light modules unfrozen (cnn_embed, post_pos_embed, Adapter, rel_pos 2/5/8/11, upneck) + mask decoder = **43.71M trainable**
- AdamW lr 1e-4, cosine over 50 epochs, bs 16 (the pre-registered light fine-tuning protocol; documented deviation from the SAMUS *official* recipe in DATA_QUALITY.md §3)

## Decision

Per the writing-team rule: the basis reads as "official weights + light fine-tuning" → **no retraining**. The reported 43.71M arm is treated as matching the registered intent; the early decoder-only 4.06M variant is archived as a rejected variant.

## Pending re-check

When OSF is reachable, re-extract the SAMUS paragraph from REGISTRATION_DOCUMENT_EN.md. If it states a concrete configuration that differs from the reported arm, run the seed-42 point check in the same format as the USFM item and update this record.

Provenance: retrieval attempts logged 2026-10-17; basis quote from plan_A v1.4 §M5 table; arm settings from scripts/m5_samus.py and DATA_QUALITY.md §3.
