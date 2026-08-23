# 1. Base Image mit vollständigem CUDA 12.4 Devel Toolkit (für nvcc und C++ Kompilierung)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Nicht-interaktive Installationen erzwingen
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Unterstützte NVIDIA GPU-Architekturen (T4, RTX 3090/4090, A100, L40S, H100)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 2. Systempakete, Python 3.10 und Headless-Rendering-Bibliotheken
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    # Headless Rendering / 3D-Bibliotheken (Trimesh, PyMeshLab, Open3D)
    libgl1-mesa-glx \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxkbcommon0 \
    libegl1 \
    libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Standard Python auf 3.10 setzen
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.10-config 1

# Pip aktualisieren
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. PyTorch mit CUDA 12.4 Unterstützung installieren
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 4. Hunyuan3D-2.1 Repository klonen
WORKDIR /app
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/Hunyuan3D-2.1

WORKDIR /app/Hunyuan3D-2.1

# 5. Python-Abhängigkeiten installieren (inkl. Blender Index für bpy)
RUN pip install --no-cache-dir pybind11
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.blender.org/pypi/

# 6. Eigene C++/CUDA Extensions kompilieren
# A) Custom Rasterizer
WORKDIR /app/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
RUN pip install -e .

# B) Differentiable Renderer (mesh_inpaint_processor)
WORKDIR /app/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh

# 7. Hilfsmodell für Texture-Upscaling (Real-ESRGAN) vorab herunterladen
WORKDIR /app/Hunyuan3D-2.1/hy3dpaint
RUN mkdir -p ckpt && \
    wget -O ckpt/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 8. Arbeitsverzeichnis und Python-Suchpfad setzen
WORKDIR /app/Hunyuan3D-2.1
ENV PYTHONPATH="/app/Hunyuan3D-2.1:${PYTHONPATH}"

# Standard-Befehl: Startet eine interaktive Shell für Terminal- & Skriptausführung
CMD ["bash"]
