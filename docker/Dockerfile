# 1. Base Image: NVIDIA CUDA 12.4.1 Devel auf Ubuntu 22.04 (enthält nativ Python 3.10)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Systempakete, natives Python 3.10 und Headless-Rendering-Bibliotheken
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    libgl1 libglx-mesa0 libgl1-mesa-dev libglib2.0-0 \
    libgomp1 libopengl0 libegl1 libegl1-mesa-dev libgles2-mesa-dev \
    libxrender1 libsm6 libxext6 xvfb ffmpeg wget curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Standard-Symlinks für Python 3.10 setzen
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.10-config 1

# 3. Pip aktualisieren & PyTorch 2.5.1 mit CUDA 12.4 installieren
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir ninja pybind11 fast-simplification hf_xet

# 4. GPU-Zielarchitekturen für nvcc definieren (+PTX für Zukunftssicherheit)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"

# 5. Hunyuan3D klonen & Original-requirements (inkl. bpy==4.0) ohne sed installieren
WORKDIR /opt/hunyuan3d
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git . && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt --extra-index-url https://download.blender.org/pypi/

# 6. C++ & CUDA Module kompilieren und global installieren
# A) Custom Rasterizer
WORKDIR /opt/hunyuan3d/hy3dpaint/custom_rasterizer
RUN pip install --no-cache-dir --no-build-isolation .

# B) Differentiable Renderer (mesh_inpaint_processor)
WORKDIR /opt/hunyuan3d/hy3dpaint/DifferentiableRenderer
RUN chmod +x compile_mesh_painter.sh && bash compile_mesh_painter.sh

# 7. Hilfsmodell für Textur-Upscaling vorab herunterladen
WORKDIR /opt/hunyuan3d/hy3dpaint
RUN mkdir -p ckpt && \
    wget -c -O ckpt/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 8. Umgebungsvariablen & Arbeitsverzeichnis für RunPod
ENV HF_HOME=/workspace/hf_cache
ENV PYOPENGL_PLATFORM=egl
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH="/opt/hunyuan3d:${PYTHONPATH}"

# Startet auf dem RunPod Network-Volume
WORKDIR /workspace

CMD ["bash"]
