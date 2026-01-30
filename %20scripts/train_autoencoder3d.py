(This includes the fixes you needed: correct batch handling + safe padding/cropping to multiple-of-8 so shapes match.)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.dataset_gli import GLIDataset

# -------------------------
# Model
# -------------------------
class Autoencoder3D(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        # Encoder: downsample 3 times (x8)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv3d(base, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        self.down1 = nn.Conv3d(base, base * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Conv3d(base * 2, base * 4, 4, stride=2, padding=1)

        self.enc3 = nn.Sequential(
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )
        self.down3 = nn.Conv3d(base * 4, base * 8, 4, stride=2, padding=1)

        self.mid = nn.Sequential(
            nn.GroupNorm(8, base * 8),
            nn.SiLU(),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.GroupNorm(8, base * 8),
            nn.SiLU(),
        )

        # Decoder
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 4, stride=2, padding=1)
        self.dec3 = nn.Sequential(
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
            nn.Conv3d(base * 4, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )

        self.up1 = nn.ConvTranspose3d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv3d(base, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )

        self.out = nn.Conv3d(base, in_ch, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.enc3(x)
        x = self.down3(x)
        x = self.mid(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.out(x)
        return x


# -------------------------
# Helpers (pad/crop to match)
# -------------------------
def pad_to_multiple(x, multiple=8):
    # x: (B,C,H,W,D)
    B, C, H, W, D = x.shape
    def _pad(n):
        r = n % multiple
        return 0 if r == 0 else (multiple - r)
    ph, pw, pd = _pad(H), _pad(W), _pad(D)
    # pad only at the end to keep alignment simple
    x = F.pad(x, (0, pd, 0, pw, 0, ph))
    return x, (H, W, D)

def crop_back(x, orig_shape):
    H, W, D = orig_shape
    return x[..., :H, :W, :D]


def save_ckpt(model, outdir, step, name=None):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name if name else f"ae3d_step{step}.pt")
    torch.save({"model": model.state_dict()}, path)
    print("saved:", path)


# -------------------------
# Train
# -------------------------
def main():
    data_root = os.path.expanduser(os.environ.get("MRI_DATA_ROOT", ""))
    outdir = os.path.expanduser(os.environ.get("OUTDIR", "./runs/ae3d_run1"))
    assert data_root, "Set MRI_DATA_ROOT"
    os.makedirs(outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = GLIDataset(data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    model = Autoencoder3D(in_ch=4, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    max_steps = int(os.environ.get("MAX_STEPS", "1250"))
    save_every = int(os.environ.get("SAVE_EVERY", "200"))

    step = 0
    epoch = 0
    model.train()

    while step < max_steps:
        for batch in loader:
            # batch = (x, patient_id)
            x = batch[0].to(device)  # (B,4,H,W,D)

            # ensure shapes are multiple-of-8 so decoder output matches
            x_pad, orig_shape = pad_to_multiple(x, multiple=8)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                y_pad = model(x_pad)
                y = crop_back(y_pad, orig_shape)
                loss = loss_fn(y, x)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")

            if step > 0 and step % save_every == 0:
                save_ckpt(model, outdir, step)

            step += 1
            if step >= max_steps:
                break

        epoch += 1

    save_ckpt(model, outdir, step, name="ae3d_last.pt")
    print("✅ done. saved:", os.path.join(outdir, "ae3d_last.pt"))


if __name__ == "__main__":
    main()
