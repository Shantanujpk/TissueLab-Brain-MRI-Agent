import os
import glob
import numpy as np
import nibabel as nib

MODS = ["t1n", "t1c", "t2w", "t2f"]
ALT_MODS = ["t1", "t1ce", "t2", "flair"]

def stats(arr):
    a = arr.astype(np.float32)
    nz = a[a > 0]
    if nz.size == 0:
        return dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean()), std=float(a.std()),
                    p1=0.0, p50=0.0, p99=0.0, nonzero_pct=0.0)
    return dict(
        min=float(a.min()),
        max=float(a.max()),
        mean=float(a.mean()),
        std=float(a.std()),
        p1=float(np.percentile(nz, 1)),
        p50=float(np.percentile(nz, 50)),
        p99=float(np.percentile(nz, 99)),
        nonzero_pct=float((nz.size / a.size) * 100.0),
    )

def find_file(patient_dir, mod):
    patt = os.path.join(patient_dir, f"*-{mod}.nii*")
    hits = sorted(glob.glob(patt))
    return hits[0] if hits else None

def describe(mod, fpath):
    img = nib.load(fpath)
    vol = img.get_fdata().astype(np.float32)
    s = stats(vol)
    print(f"\n[{mod}] ✅ found: {fpath}")
    print(" affine shape:", img.affine.shape)
    print("  shape     :", vol.shape)
    print("  dtype     :", vol.dtype)
    print("  min       :", s["min"])
    print("  max       :", s["max"])
    print("  mean      :", s["mean"])
    print("  std       :", s["std"])
    print("  p1        :", s["p1"])
    print("  p50       :", s["p50"])
    print("  p99       :", s["p99"])
    print("  nonzero_% :", s["nonzero_pct"])

def main():
    root = os.path.expanduser(os.environ.get("MRI_DATA_ROOT", ""))
    pid = os.environ.get("PATIENT_ID", "")
    assert root and pid, "Set MRI_DATA_ROOT and PATIENT_ID"

    pdir = os.path.join(root, pid)
    print("=== MRI Realism: Sanity Check ===")
    print("DATA_ROOT:", root)
    print("PATIENT_DIR:", pdir)

    print("\n--- Searching using MODS:", MODS, "---")
    for mod in MODS:
        f = find_file(pdir, mod)
        if f:
            describe(mod, f)
        else:
            print(f"\n[{mod}]  ❌ not found")

    print("\n--- Searching using ALT_MODS:", ALT_MODS, "---")
    for mod in ALT_MODS:
        f = find_file(pdir, mod)
        if f:
            describe(mod, f)
        else:
            print(f"[{mod}]  ❌ not found")

if __name__ == "__main__":
    main()
