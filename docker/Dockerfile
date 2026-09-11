# ==============================================================================
# STAGE 1: Builder (Kompilierung mit nvcc, C++ Build-Tools & uv)
# ==============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"

# Robuster System-Paket-Download (IPv4-Zwang + 30s Timeout gegen Hänger)
RUN apt-get update -o Acquire::ForceIPv4=true -o Acquire::http::Timeout="30" && \
    apt-get install -y --no-install-recommends \
    -o Acquire::ForceIPv4=true -o Acquire::http::Timeout="30" \
    git build-essential cmake ninja-build \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    libgl1 libglx-mesa0 libgl1-mesa-dev libglib2.0-0 \
    libgomp1 libopengl0 libegl1 libegl1-mesa-dev libgles2-mesa-dev \
    wget curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Symlinks für Python 3.10 setzen
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.10-config 1

# uv installieren (beschleunigt nachfolgende Python-Downloads massiv)
RUN python3 -m pip install --no-cache-dir --upgrade pip uv

# PyTorch-Stack (inkl. torchaudio) via uv parallel herunterladen
RUN uv pip install --system \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --system ninja pybind11 fast-simplification hf_xet

# Hunyuan3D klonen, Abhängigkeiten parallel laden & Extensions bauen
WORKDIR /opt/hunyuan3d
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git . && \
    rm -rf .git && \
    # Requirements via uv parallel installieren (inkl. Blender bpy)
    uv pip install --system --no-build-isolation -r requirements.txt --extra-index-url https://download.blender.org/pypi/ && \
    # A) Custom Rasterizer kompilieren & temporären build/-Ordner sofort löschen
    cd /opt/hunyuan3d/hy3dpaint/custom_rasterizer && \
    pip install --no-cache-dir --no-build-isolation . && \
    rm -rf build *.egg-info && \
    # B) Differentiable Renderer kompilieren
    cd /opt/hunyuan3d/hy3dpaint/DifferentiableRenderer && \
    chmod +x compile_mesh_painter.sh && bash compile_mesh_painter.sh && \
    # RealESRGAN Hilfsmodell vorab laden
    mkdir -p /opt/hunyuan3d/hy3dpaint/ckpt && \
    wget -c -O /opt/hunyuan3d/hy3dpaint/ckpt/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth


# ==============================================================================
# STAGE 2: Finales Runtime-Image für RunPod (Schlank & optimiert)
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Laufzeitbibliotheken, Git & SSH installieren
RUN apt-get update -o Acquire::ForceIPv4=true -o Acquire::http::Timeout="30" && \
    apt-get install -y --no-install-recommends \
    -o Acquire::ForceIPv4=true -o Acquire::http::Timeout="30" \
    git \
    python3.10 python3.10-distutils python3-pip \
    openssh-server \
    libgl1 libglx-mesa0 libglib2.0-0 libgomp1 libopengl0 libegl1 \
    libxrender1 libsm6 libxext6 xvfb ffmpeg wget curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Symlinks für Python 3.10 setzen
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# SSH-Dienst vorkonfigurieren (Root-Login erlauben)
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Fertig installierte Python-Pakete & kompilierten Code aus Stage 1 übernehmen
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/hunyuan3d /opt/hunyuan3d

# Eigene Skripte kopieren & global verlinken
COPY scripts/batch_shapegen.py scripts/batch_texturegen.py /opt/hunyuan3d/
RUN chmod +x /opt/hunyuan3d/batch_shapegen.py /opt/hunyuan3d/batch_texturegen.py && \
    ln -s /opt/hunyuan3d/batch_shapegen.py /usr/local/bin/batch_shapegen && \
    ln -s /opt/hunyuan3d/batch_texturegen.py /usr/local/bin/batch_texturegen

# Laufzeit-Umgebungsvariablen
ENV HUNYUAN_ROOT=/opt/hunyuan3d
ENV HF_HOME=/workspace/hf_cache
ENV PYOPENGL_PLATFORM=egl
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH="/opt/hunyuan3d:/opt/hunyuan3d/hy3dshape:/opt/hunyuan3d/hy3dpaint"

# SSH-Port freigeben
EXPOSE 22

WORKDIR /workspace

# Start-Kommando für RunPod (SSH-Key hinterlegen, SSH starten, Pod aktiv halten)
CMD ["sh", "-c", "[ -n \"$PUBLIC_KEY\" ] && echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys; service ssh start && sleep infinity"]