# Stable-DragGan (Stability & Tracking Enhanced Version of Drag Your Gan)

## Enhanced Features in this Fork

This repository improves upon the original DragGAN implementation by introducing advanced regularization techniques to prevent artifacts and latent space drifting during long-range drags:

1.  **Codebook Regularization (VQ-Constraint):**
    * Utilizes K-Means clustering on the $W$ latent space to create a "Codebook" of valid latent representations.
    * Applies a Vector Quantization (VQ) loss during optimization to ensure the latent codes stay close to the distribution of real images.
    * Supports **Adaptive Weights** and **Multi-scale** constraints for $W+$ space.

2.  **PCA Gradient Projection:**
    * Projects the optimization gradients onto the Principal Components (PCA) of the latent space.
    * Ensures that edits move along semantically valid directions, significantly reducing high-frequency artifacts.

3.  **Robust Tracking (Patch-based):**
    * Implements a new `Robust` tracking mode using patch matching ($5 \times 5$) and spatial inertia.
    * Prevents handle points from drifting in texture-less regions (e.g., sky, skin).

---

## Requirements

If you have CUDA graphic card, please follow the requirements of [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3#requirements).  

The usual installation steps involve the following commands:

```bash
conda env create -f environment.yml
conda activate stylegan3
pip install -r requirements.txt
```

**Additional Requirements for Enhanced Features:** You must install scikit-learn, matplotlib and umap-learn to run the codebook and PCA generation scripts:

```bash
pip install scikit-learn matplotlib umap-learn
```

Otherwise (for GPU acceleration on MacOS with Silicon Mac M1/M2, or just CPU) try the following:

```bash
cat environment.yml | \
  grep -v -E 'nvidia|cuda' > environment-no-nvidia.yml && \
    conda env create -f environment-no-nvidia.yml
conda activate stylegan3

# On MacOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Usage Workflow

To fully utilize the enhanced features, you need to pre-compute the Codebook and PCA components for your specific model (e.g., Lions, Human, etc.).

1. **Download Pre-trained Weights**

```bash
python scripts/download_model.py
```

2. **Generate Codebook (Offline)**
Run the analysis script to sample latent vectors and cluster them. This creates a .npy file used for regularization.

```bash
# Example for Lions model
python generate_codebook.py \
    --network checkpoints/stylegan2_lions_512_pytorch.pkl \
    --output checkpoints/codebook_lions.npy \
    --auto
```

  --auto: Automatically determines optimal codebook size and sample count based on model resolution.

  --device: Specify cuda or mps.

3. **Generate PCA Components (Offline)**
Compute PCA components based on the generated codebook to enable Gradient Projection.

```bash
python generate_pca.py \
    --codebook checkpoints/codebook_lions.npy \
    --output pca_components.npy \
```

4. **Run the Enhanced DragGAN GUI**
Start the visualizer. The UI has been updated to accept these new files.

```bash
python visualizer_drag_gradio.py
```

## GUI Controls Guide

The Gradio interface now includes specific panels for the new features:

**Tracking Robustness (Experiment)**

  - **Tracking Algorithm**: Switch between Standard (Original Pixel-wise) and Robust (Patch+Inertia).

    - Use Robust mode for smoother dragging in smooth/plain areas.

  - **Patch Size**: Adjust the size of the feature patch (default: 5).

  - **Spatial Penalty**: Controls how much the point resists jumping long distances.

**VQ Constraints (Codebook)**

  - **Upload Codebook**: Upload the .npy file generated in Step 2.

  - **VQ Mode**:

    - **Static**: Fixed weight regularization.

    - **Adaptive**: Changes weight based on drag distance (higher weight when dragging far).

  - **Base VQ Weight**: Strength of the pull towards valid clusters.

**PCA Projection**

  **Enable PCA Gradient Projection**: Toggle this on to constrain updates to principal components.

  **Load PCA**: Click to load the .npy file generated in Step 3 (or it autoloads if named correctly).

## Run Gradio visualizer in Docker

Provided docker image is based on NGC PyTorch repository. To quickly try out visualizer in Docker, run the following:

```bash
# before you build the docker container, make sure you have cloned this repo, and downloaded the pretrained model by `python scripts/download_model.py`.
docker build . -t draggan:latest  
docker run -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash
# (Use GPU)if you want to utilize your Nvidia gpu to accelerate in docker, please add command tag `--gpus all`, like:
#   docker run --gpus all  -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash

cd src && python visualizer_drag_gradio.py --listen
```

Now you can open a shared link from Gradio (printed in the terminal console).

Beware the Docker image takes about 25GB of disk space!

## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3). Part of the code is borrowed from [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human).

(cheers to the community as well)

## License

The code related to the DragGAN algorithm is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/). However, most of this project are available under a separate license terms: all codes used or modified from [StyleGAN3](https://github.com/NVlabs/stylegan3) is under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

Any form of use and derivative of this code must preserve the watermarking functionality showing "AI Generated".

## BibTeX

```bash
@inproceedings{pan2023draggan,
    title={Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold},
    author={Pan, Xingang and Tewari, Ayush, and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    year={2023}
}
```