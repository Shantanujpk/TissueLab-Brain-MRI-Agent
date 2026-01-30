# MRI Realism (3D Autoencoder Baseline)

This repo trains a simple 3D convolutional autoencoder on multi-modal BraTS GLI MRI volumes
(T1n, T1c, T2w, T2f). The goal is to learn a compact 3D latent representation and verify
reconstruction quality before moving to a generative model (e.g., diffusion in latent space)
for realistic MRI synthesis.

## Data Expected
BraTS 2023 GLI training data layout:
${MRI_DATA_ROOT}/BraTS-GLI-00002-000/
  BraTS-GLI-00002-000-t1n.nii.gz
  BraTS-GLI-00002-000-t1c.nii.gz
  BraTS-GLI-00002-000-t2w.nii.gz
  BraTS-GLI-00002-000-t2f.nii.gz

## Environment (GPU server)
We used a Conda environment with CUDA-enabled PyTorch.

### Load conda in /bin/sh shells
If conda is not found:
```bash
. ~/miniconda3/etc/profile.d/conda.sh
--------------------------------------------------------------------------------------------------------
Create env + install packages
conda create -n mri_realism python=3.10 -y
conda activate mri_realism

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install nibabel matplotlib
--------------------------------------------------------------------------------------------------------
GPU Selection
export CUDA_VISIBLE_DEVICES=3
----------------------------------------------------------------------------------------------------------
Step 1 — Sanity-check a patient
cd ~/sjaipurkar/projects/mri_realism
export PYTHONPATH=$(pwd)

export MRI_DATA_ROOT=~/sjaipurkar/data/GLI/training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
export PATIENT_ID=BraTS-GLI-00002-000

python scripts/check_gli_sample.py

------------------------------------------------------------------
Step 2 — Train 3D Autoencoder
cd ~/sjaipurkar/projects/mri_realism
export PYTHONPATH=$(pwd)

export CUDA_VISIBLE_DEVICES=3
export MRI_DATA_ROOT=~/sjaipurkar/data/GLI/training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
export OUTDIR=~/sjaipurkar/projects/mri_realism/runs/ae3d_run1

python scripts/train_autoencoder3d.py
Checkpoints saved to:

${OUTDIR}/ae3d_step*.pt
${OUTDIR}/ae3d_last.pt
-----------------------------------------------------------------------------------------------------
Step 3 — Visualize reconstruction (Original vs Recon)
cd ~/sjaipurkar/projects/mri_realism
export PYTHONPATH=$(pwd)

export CUDA_VISIBLE_DEVICES=3
export MRI_DATA_ROOT=~/sjaipurkar/data/GLI/training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
export OUTDIR=~/sjaipurkar/projects/mri_realism/runs/ae3d_run1
export PATIENT_ID=BraTS-GLI-00002-000

python scripts/viz_ae3d_recon.py

Output example:
runs/ae3d_run1/ae3d_recon_BraTS-GLI-00002-000.png
Repeat for another patient:
BraTS-GLI-00030-000


