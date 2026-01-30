(This version imports the correct class from train_autoencoder3d.py.)

import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

MODS = ["t1n", "t1c", "t2w", "t2f"]

def find_mod_file(patient_dir, mod):
    import glob
    patt = os.path.join(patient_dir, f"*-{mod}.nii*")
    hits = sorted(glob.glob(patt))
    return hits[0] if hits else None

def robust_normalize(vol):
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

def pad_to_multiple(x, multiple=8):
    import torch.nn.functional as F
    B, C, H, W, D = x.shape
    def _pad(n):
        r = n % multiple
        return 0 if r == 0 else (multiple - r)
    ph, pw, pd = _pad(H), _pad(W), _pad(D)
    x = F.pad(x, (0, pd, 0, pw, 0, ph))
    return x, (H, W, D)

def crop_back(x, orig_shape):
    H, W, D = orig_shape
    return x[..., :H, :W, :D]

def load_patient_tensor(root, pid):
    pdir = os.path.join(root, pid)
    vols = []
    for mod in MODS:
        f = find_mod_file(pdir, mod)
        if f is None:
            raise FileNotFoundError(f"Missing {mod} for {pid}")
        vol = nib.load(f).get_fdata().astype(np.float32)
        vol = robust_normalize(vol)
        vols.append(vol)
    x = np.stack(vols, axis=0)  # (4,H,W,D)
    x = torch.from_numpy(x).unsqueeze(0).float()  # (1,4,H,W,D)
    return x, pdir

def load_model(ckpt_path, device):
    # Model class is defined in train_autoencoder3d.py
    from scripts.train_autoencoder3d import Autoencoder3D
    model = Autoencoder3D(in_ch=4, base=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def save_recon_figure(x, y, out_png):
    # x,y: (1,4,H,W,D)
    x = x[0].detach().cpu().numpy()
    y = y[0].detach().cpu().numpy()
    # mid slice along D
    mid = x.shape[-1] // 2

    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    for i, mod in enumerate(MODS):
        axes[i, 0].imshow(x[i, :, :, mid], cmap="gray")
        axes[i, 0].set_title(f"{mod} - Original (mid slice)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(y[i, :, :, mid], cmap="gray")
        axes[i, 1].set_title(f"{mod} - Recon (mid slice)")
        axes[i, 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("saved:", out_png)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    root = os.path.expanduser(os.environ.get("MRI_DATA_ROOT", ""))
    outdir = os.path.expanduser(os.environ.get("OUTDIR", "./runs/ae3d_run1"))
    pid = os.environ.get("PATIENT_ID", "BraTS-GLI-00002-000")
    ckpt_path = os.path.join(outdir, "ae3d_last.pt")

    print("CKPT:", ckpt_path)
    print("PATIENT:", pid)

    x, _ = load_patient_tensor(root, pid)
    x = x.to(device)

    model = load_model(ckpt_path, device)

    with torch.no_grad():
        x_pad, orig_shape = pad_to_multiple(x, multiple=8)
        y_pad = model(x_pad)
        y = crop_back(y_pad, orig_shape)

    out_png = os.path.join(outdir, f"ae3d_recon_{pid}.png")
    save_recon_figure(x, y, out_png)

if __name__ == "__main__":
    main()
