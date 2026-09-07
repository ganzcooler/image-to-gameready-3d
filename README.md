# image-to-gameready-3d

GPU-powered batch conversion from reference images to game-ready 3D assets.
This repository packages [Tencent Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
in a CUDA-enabled Docker image and adds command-line batch scripts for shape
generation and texture generation.

## Pipeline

The workflow can be run in two stages:

1. Generate an untextured mesh from each input image with `batch_shapegen`.
2. Generate a textured `.glb` from each mesh and its matching reference image
   with `batch_texturegen`.

The image includes PyTorch 2.5.1 for CUDA 12.4, Blender Python (`bpy`), the
Hunyuan3D native extensions, EGL-based headless rendering, and the
RealESRGAN upscaling checkpoint used by the texturing pipeline.

## Requirements

- Docker with NVIDIA Container Toolkit
- An NVIDIA GPU compatible with the configured CUDA build
- Enough GPU memory for the selected Hunyuan3D settings
- A Hugging Face cache or network access for the Hunyuan3D model weights

The Dockerfile targets NVIDIA CUDA 12.4.1, Ubuntu 22.04, and Python 3.10.
The image is large and the first build downloads several model and dependency
artifacts.

## Build and publish the image

Run this command from the repository root. The Docker build context must be
the `docker` directory because the Dockerfile copies the batch scripts from
`docker/scripts`.

```bash
export GHCR_IMAGE="ghcr.io/<github-owner>/image-to-gameready-3d"

docker build -t "${GHCR_IMAGE}:latest" ./docker
```

Authenticate with GitHub Container Registry using a GitHub token that has
permission to write packages, then push the image:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
docker push "${GHCR_IMAGE}:latest"
```

Replace `<github-owner>` and `<github-username>` with the GitHub account or
organization that owns the package. The image can then be selected as a
custom container image when creating a RunPod pod:

```text
ghcr.io/<github-owner>/image-to-gameready-3d:latest
```

The package must be public, or the RunPod environment must be authenticated to
pull private GHCR images. Mount or upload your input files under `/workspace`
when configuring the pod. The image already starts the required container
process and installs `batch_shapegen` and `batch_texturegen` globally.

## Generate 3D shapes

Create a RunPod pod using the GHCR image above and place supported images in an
input directory such as `/workspace/inputs`. Supported formats are PNG, JPG,
JPEG, WebP, BMP, and TIFF. Open a terminal inside the running pod and execute:

```bash
batch_shapegen \
  --input_dir /workspace/inputs \
  --output_dir /workspace/outputs_shapes \
  --format glb \
  --skip_existing
```

Each input image produces a mesh with the same filename stem. The default
output format is `glb`; `obj` is also supported.

Useful options include:

| Option | Default | Description |
| --- | --- | --- |
| `--model_path` | `tencent/Hunyuan3D-2.1` | Hugging Face model ID or local model path |
| `--format` | `glb` | Output format: `glb` or `obj` |
| `--octree_resolution` | `256` | Mesh resolution; higher values require more resources |
| `--num_inference_steps` | model default | Diffusion inference steps |
| `--guidance_scale` | model default | Guidance scale |
| `--seed` | unset | Reproducible generation seed |
| `--no_rembg` | disabled | Skip automatic background removal |
| `--force_rembg` | disabled | Always run background removal |
| `--device` | `cuda` when available | Torch device |
| `--skip_existing` | disabled | Do not regenerate existing output files |

For example, to generate OBJ files reproducibly:

```bash
batch_shapegen \
  -i /workspace/inputs \
  -o /workspace/outputs_shapes \
  --format obj \
  --seed 42
```

## Generate textured models

`batch_texturegen` reads untextured OBJ files and finds reference images by
matching filenames. For example:

```text
/workspace/meshes/chair.obj
/workspace/reference-images/chair.png
```

Create the RunPod pod with the same GHCR image, ensure the OBJ files and
reference images are available under `/workspace`, then run the command from a
terminal inside the pod:

```bash
batch_texturegen \
  --input_dir /workspace/meshes \
  --image_dir /workspace/reference-images \
  --output_dir /workspace/outputs_textured \
  --skip_existing
```

The result is `/workspace/outputs_textured/chair.glb`. If no matching image
exists, use `--fallback_image` to provide a shared fallback image. Supported
reference image formats are PNG, JPG, JPEG, WebP, and BMP.

Useful options include:

| Option | Default | Description |
| --- | --- | --- |
| `--max_num_view` | `6` | Number of generated views, from 6 to 9 |
| `--resolution` | `512` | Texture resolution: `512` or `768` |
| `--no_rembg` | disabled | Disable automatic background removal |
| `--fallback_image` | unset | Image used when no stem-matched image exists |
| `--skip_existing` | disabled | Do not regenerate existing `.glb` files |

Example with higher texture resolution:

```bash
batch_texturegen \
  -i /workspace/meshes \
  --image_dir /workspace/reference-images \
  -o /workspace/outputs_textured \
  --resolution 768
```

## Validate an image

After building the image, run the included validation script from the
repository root:

```bash
bash docker/test_image.sh "${GHCR_IMAGE}:latest"
```

The script checks Python, PyTorch, CUDA bindings, Blender, mesh-processing
libraries, Hunyuan3D pipeline imports, native extensions, the RealESRGAN
checkpoint, and basic upstream script parsing.

The validation script uses a CPU Docker invocation for import and integrity
checks. It does not prove that a full generation job will fit in a particular
GPU's memory.

## Configuration

The image sets these environment variables:

| Variable | Value | Purpose |
| --- | --- | --- |
| `HUNYUAN_ROOT` | `/opt/hunyuan3d` | Installed Hunyuan3D root |
| `HF_HOME` | `/workspace/hf_cache` | Hugging Face model cache |
| `PYOPENGL_PLATFORM` | `egl` | Headless OpenGL backend |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA installation path |

To persist downloaded Hugging Face models between RunPod sessions, attach a
persistent volume to the pod and mount it at `/workspace/hf_cache`. This keeps
the model cache aligned with the `HF_HOME` value without rebuilding the image.

## Project layout

```text
docker/
├── Dockerfile
├── test_image.sh
└── scripts/
    ├── batch_shapegen.py
    └── batch_texturegen.py
```
