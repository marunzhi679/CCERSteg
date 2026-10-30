# CCER-Steg: Towards Imperceptible Remote Sensing Image Steganography via Content-Coordinated Embedding and Information-Guided Routing

**Runzhi Ma, Wenqian Dong, Song Xiao, Jiahui Qu, Yunsong Li**

The PyTorch implementation of **Content-Coordinated Expert-Routing Steganography (CCER-Steg)** for remote sensing image steganography.

**Paper:** Coming soon.

## Abstract

Remote sensing (RS) image steganography aims to hide secret images in a cover image while maintaining stego-image imperceptibility and accurate secret recovery. However, existing image-hiding methods may not sufficiently account for regional differences and complementary information in RS imagery. To address these challenges, we propose **CCER-Steg**, a content-coordinated expert-routing framework. Specifically, **Structure-Aware Dynamic Allocation (SADA)** adaptively determines secret allocation using local structural characteristics and cover–secret interactions. **Structure-Conditioned Positional Encoding (SCPE)** helps preserve spatial relationships during feature processing. **Multi-Dimensional Information-Guided Mixture-of-Experts (MDI-MoE)** exploits local gradients, inter-channel covariance, and spatial context to guide expert selection. Experiments on remote sensing datasets demonstrate high stego-image fidelity and accurate secret reconstruction. The framework also supports single-image, multi-image, and RGB–IR cross-modal steganography.

## Requirements

Install PyTorch and the required Python packages for your environment, including `torchvision`, `numpy`, `opencv-python`, `Pillow`, `tqdm`, `pytorch-msssim`, and `einops`.

## How to Train

- Prepare the training datasets and update the dataset paths and training settings in the corresponding configuration files.
- Run the training script for the desired hiding task:

```bash
# Single RGB secret
python train_single.py

# Two RGB secrets
python train_rgb2.py

# Three RGB secrets
python train_rgb3.py

# RGB + IR secrets
python train_multi.py
```

The corresponding configuration files are `config.py`, `config_rgb2.py`, `config_rgb3.py`, and `config_multi.py`, respectively.

## How to Test

- Pretrained weights for the different tasks are provided in the `checkpoint/` folder.
- Set the dataset paths and the matching checkpoint path (`CKPT_PATH`) in the test script.
- Run the corresponding test script:

```bash
# Single RGB secret
python test_single.py

# Two or three RGB secrets
python test_multiRGB.py

# RGB + IR secrets
python test_multi.py
```

For two- or three-image hiding, select the corresponding `config_rgb2.py` or `config_rgb3.py` in `test_multiRGB.py`. The test scripts report PSNR, SSIM, MAE, and RMSE and save the generated images.

## Pretrained Models

The pretrained checkpoints for single-RGB, two-RGB, three-RGB, and RGB–IR steganography are stored in `checkpoint/`. Please use the checkpoint corresponding to the selected task.

## Contact

If you have any questions, please open an issue in this repository.

## Citation

If you find this work useful, please cite our manuscript. The following entry is provisional and will be updated upon publication:

