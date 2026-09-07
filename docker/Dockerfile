# 1. Base Image: NVIDIA CUDA 12.4.1 Devel auf Ubuntu 22.04 (nativ Python 3.10)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Systempakete, OpenSSH-Server (für SCP/SFTP) und Headless-Rendering-Bibliotheken
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    openssh-server \
    libgl1 libglx-mesa0 libgl1-mesa-dev libglib2.0-0 \
    libgomp1 libopengl0 libegl1 libegl1-mesa-dev libgles2-mesa-dev \
    libxrender1 libsm6 libxext6 xvfb ffmpeg wget curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Standard-Symlinks für Python 3.10 setzen
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.10-config 1

# 3. SSH-Dienst vorkonfigurieren (Root-Login & Berechtigungen)
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 4. Pip aktualisieren & PyTorch 2.5.1 mit CUDA 12.4 installieren
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir ninja pybind11 fast-simplification hf_xet

# 5. GPU-Zielarchitekturen für nvcc definieren (+PTX für Zukunftssicherheit)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"

# 6. Hunyuan3D klonen & Original-requirements (inkl. bpy==4.0) ohne sed installieren
WORKDIR /opt/hunyuan3d
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git . && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt --extra-index-url https://download.blender.org/pypi/

# 7. C++ & CUDA Erweiterungen fest im System installieren
# A) Custom Rasterizer
WORKDIR /opt/hunyuan3d/hy3dpaint/custom_rasterizer
RUN pip install --no-cache-dir --no-build-isolation .

# B) Differentiable Renderer (mesh_inpaint_processor)
WORKDIR /opt/hunyuan3d/hy3dpaint/DifferentiableRenderer
RUN chmod +x compile_mesh_painter.sh && bash compile_mesh_painter.sh

# 8. Hilfsmodell für Textur-Upscaling vorab herunterladen
WORKDIR /opt/hunyuan3d/hy3dpaint
RUN mkdir -p ckpt && \
    wget -c -O ckpt/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 8b. Eigene Batch-Skripte aus dem Unterordner "scripts/" kopieren & global verlinken
COPY scripts/batch_shapegen.py scripts/batch_texturegen.py /opt/hunyuan3d/
RUN chmod +x /opt/hunyuan3d/batch_shapegen.py /opt/hunyuan3d/batch_texturegen.py && \
    ln -s /opt/hunyuan3d/batch_shapegen.py /usr/local/bin/batch_shapegen && \
    ln -s /opt/hunyuan3d/batch_texturegen.py /usr/local/bin/batch_texturegen

# 9. Umgebungsvariablen & Laufzeitpfade
ENV HUNYUAN_ROOT=/opt/hunyuan3d
ENV HF_HOME=/workspace/hf_cache
ENV PYOPENGL_PLATFORM=egl
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH="/opt/hunyuan3d:${PYTHONPATH}"

# SSH-Port freigeben
EXPOSE 22

# Start-Arbeitsverzeichnis
WORKDIR /workspace

# 10. Startbefehl: SSH-Key übernehmen, SSH-Server starten & Container am Leben halten
CMD ["sh", "-c", "[ -n \"$PUBLIC_KEY\" ] && echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys; service ssh start && sleep infinity"]