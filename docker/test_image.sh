#!/bin/bash

# Bildname als Parameter (Standard: hunyuan3d:latest)
IMAGE_NAME="${1:-hunyuan3d:latest}"

echo "============================================================"
echo "  Starte CPU-Testsuite für Image: $IMAGE_NAME"
echo "============================================================"

docker run -i --rm "$IMAGE_NAME" python3 - << 'EOF'
import sys
import os

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

failed_tests = 0

def check(name, test_func):
    global failed_tests
    try:
        result = test_func()
        print(f"[{GREEN}OK{RESET}] {name}: {result if result is not None else 'OK'}")
    except Exception as e:
        failed_tests += 1
        print(f"[{RED}FAIL{RESET}] {name}: FEHLER -> {e}")

print("\n--- 1. Python & Core Frameworks ---")
check("Python Version", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}" if (sys.version_info.major == 3 and sys.version_info.minor == 10) else (_ for _ in ()).throw(ValueError(f"Falsche Version: {sys.version}")))

import torch
check("PyTorch Version", lambda: torch.__version__)
check("PyTorch CUDA-Anbindung", lambda: f"CUDA {torch.version.cuda} Build")

import torchvision
check("Torchvision Version", lambda: torchvision.__version__)

import torchaudio
check("Torchaudio Version", lambda: torchaudio.__version__)

print("\n--- 2. 3D- & Rendering-Bibliotheken ---")
import bpy
check("Blender (bpy)", lambda: f"v{bpy.app.version_string}")

import trimesh
check("Trimesh", lambda: f"v{trimesh.__version__}")

import pymeshlab
from importlib.metadata import version
check("PyMeshLab", lambda: f"v{version('pymeshlab')}")

import cv2
check("OpenCV", lambda: f"v{cv2.__version__}")

print("\n--- 3. Eigene C++/CUDA Extensions (Kritisch!) ---")
import custom_rasterizer
check("Custom Rasterizer (C++/CUDA .so)", lambda: "Erfolgreich geladen")

sys.path.append("/opt/hunyuan3d/hy3dpaint/DifferentiableRenderer")
import mesh_inpaint_processor
check("Mesh Inpaint Processor (C++ .so)", lambda: "Erfolgreich geladen")

print("\n--- 4. Hunyuan3D Codebase & Pipeline-Klassen ---")
sys.path.extend(["/opt/hunyuan3d", "/opt/hunyuan3d/hy3dshape", "/opt/hunyuan3d/hy3dpaint"])

import hy3dshape
check("Hunyuan3D Shape Core (hy3dshape)", lambda: "Erfolgreich geladen")

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
check("ShapeGen Pipeline Klasse", lambda: Hunyuan3DDiTFlowMatchingPipeline.__name__)

from textureGenPipeline import Hunyuan3DPaintPipeline
check("Paint/Texture Pipeline Klasse", lambda: Hunyuan3DPaintPipeline.__name__)

print("\n--- 5. Hilfsdateien & Checkpoints ---")
ckpt_path = "/opt/hunyuan3d/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
def check_ckpt():
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {ckpt_path}")
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    if size_mb < 50:
        raise ValueError(f"Datei unvollständig ({size_mb:.2f} MB)")
    return f"{size_mb:.2f} MB vorhanden"
check("RealESRGAN Checkpoint", check_ckpt)

print("\n------------------------------------------------------------")
sys.stdout.flush()
sys.stderr.flush()
if failed_tests == 0:
    print(f"{GREEN}🎉 ALLE PYTHON-TESTS BESTANDEN!{RESET}")
    os._exit(0)
else:
    print(f"{RED}❌ {failed_tests} TEST(S) FEHLGESCHLAGEN!{RESET}")
    os._exit(1)
EOF

PYTHON_EXIT_CODE=$?

echo ""
echo "--- 6. Eigene Batch-Skripte & Symlinks prüfen ---"
# Prüft deine Skripte auf Syntaxfehler
for script in batch_shapegen.py batch_texturegen.py; do
    docker run --rm "$IMAGE_NAME" python3 -m py_compile "/opt/hunyuan3d/$script" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "[\033[92mOK\033[0m] /opt/hunyuan3d/$script syntaktisch fehlerfrei"
    else
        echo -e "[\033[91mFAIL\033[0m] /opt/hunyuan3d/$script hat Syntaxfehler oder fehlt"
        PYTHON_EXIT_CODE=1
    fi
done

# Prüft, ob die globalen CLI-Befehle verlinkt und aufrufbar sind
for cmd in batch_shapegen batch_texturegen; do
    docker run --rm "$IMAGE_NAME" which $cmd > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "[\033[92mOK\033[0m] Globaler Befehl '$cmd' im PATH gefunden"
    else
        echo -e "[\033[91mFAIL\033[0m] Globaler Befehl '$cmd' fehlt in /usr/local/bin"
        PYTHON_EXIT_CODE=1
    fi
done

echo ""
echo "--- 7. System- & SSH-Konfiguration ---"
# Prüft die sshd_config auf Gültigkeit
docker run --rm "$IMAGE_NAME" sshd -t > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "[\033[92mOK\033[0m] SSH-Daemon Konfiguration ist gültig"
else
    echo -e "[\033[91mFAIL\033[0m] SSH-Daemon Konfiguration fehlerhaft"
    PYTHON_EXIT_CODE=1
fi

echo "============================================================"
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo -e "\033[92m✔ Bereit für RunPod! Das Image kann bedenkenlos gepusht werden.\033[0m"
else
    echo -e "\033[91m✘ Bitte prüfe die Fehlermeldungen oben im Protokoll vor dem Push.\033[0m"
    exit 1
fi