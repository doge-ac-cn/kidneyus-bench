#!/usr/bin/env python3
"""Pack scored-slice masks (GT / Annotator1 / Annotator2) into per-kidney
bit-packed npz caches (parallel, skips existing)."""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
RAW = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks"
OUT = ROOT / "outputs/masks_packed"
OUT.mkdir(parents=True, exist_ok=True)


def pack_one(job):
    kid, zs = job
    outp = OUT / f"{kid}.npz"
    if outp.exists():
        return f"{kid} skip"
    G = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / "GT_estimated_masksUS" / f"{kid}_maskUS.nii.gz"))) > 0.5
    A = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / "Annotator1" / f"{kid}1_maskUS.nii.gz"))) > 0.5
    B = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / "Annotator2" / f"{kid}2_maskUS.nii.gz"))) > 0.5
    shape = G.shape[1:]
    pg = np.stack([np.packbits(G[z].reshape(-1)) for z in zs])
    pa = np.stack([np.packbits(A[z].reshape(-1)) for z in zs])
    pb = np.stack([np.packbits(B[z].reshape(-1)) for z in zs])
    np.savez_compressed(outp, zs=np.array(zs, np.int32),
                        shape=np.array(shape, np.int32), G=pg, A=pa, B=pb)
    return f"{kid} {len(zs)}"


def main():
    sl = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
    jobs = [(kid, sorted(int(z) for z in g["slice_idx"]))
            for kid, g in sl.groupby("kidney_id", sort=True)]
    with Pool(2) as pool:
        for r in pool.imap_unordered(pack_one, jobs, chunksize=1):
            print(r, flush=True)
    print("PACK DONE", flush=True)


if __name__ == "__main__":
    main()
