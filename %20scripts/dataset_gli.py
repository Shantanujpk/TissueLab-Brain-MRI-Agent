
---

# ✅ scripts/dataset_gli.py (copy-paste)

```python
import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

MODS = ["t1n", "t1c", "t2w", "t2f"]

def _find_mod_file(patient_dir: str, mod: str):
    # BraTS naming: BraTS-GLI-XXXX-XXX-t1n.nii.gz
    patt = os.path.join(patient_dir, f"*-{mod}.nii*")
    hits = sorted(glob.glob(patt))
    return hits[0] if hits else None

def _robust_normalize(vol: np.ndarray):
    # normalize using non-zero region percentiles
    v = vol.astype(np.float32)
    nz = v[v > 0]
    if nz.size < 10:
        return v
    p1 = np.percentile(nz, 1)
    p99 = np.percentile(nz, 99)
    if p99 <= p1:
        return v
    v = np.clip(v, p1, p99)
    v = (v - p1) / (p99 - p1 + 1e-8)
    return v

class GLIDataset(Dataset):
    """
    Returns a single tensor x of shape (4, H, W, D) for each patient.
    """
    def __init__(self, root: str):
        self.root = os.path.expanduser(root)
        self.patient_dirs = sorted([
            os.path.join(self.root, d)
            for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d)) and d.startswith("BraTS-GLI-")
        ])

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        pdir = self.patient_dirs[idx]
        chans = []
        for mod in MODS:
            f = _find_mod_file(pdir, mod)
            if f is None:
                raise FileNotFoundError(f"Missing modality {mod} in {pdir}")
            img = nib.load(f)
            vol = img.get_fdata().astype(np.float32)  # (H,W,D)
            vol = _robust_normalize(vol)
            chans.append(vol)

        x = np.stack(chans, axis=0)  # (4,H,W,D)
        x = torch.from_numpy(x).float()
        return x, os.path.basename(pdir)
