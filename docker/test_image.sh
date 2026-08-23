# Image builden: docker build -t hunyuan3d:latest .
# Prüfen ob image da ist: docker images | grep hunyuan3d
# chmod +x test_image.sh
# ./test_image.sh hunyuan3d:latest (wenn image so heißt)
#!/bin/bash

# Bildname als Parameter (Standard: hunyuan3d:latest)
IMAGE_NAME="${1:-hunyuan3d:latest}"

echo "============================================================"
echo "  Starte CPU-Testsuite für Image: $IMAGE_NAME"
echo "============================================================"

docker run --rm "$IMAGE_NAME" python3 - << 'EOF'
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

print("\n--- 2. 3D- & Rendering-Bibliotheken ---")
import bpy
check("Blender (bpy)", lambda: f"v{bpy.app.version_string}")

import trimesh
check("Trimesh", lambda: f"v{trimesh.__version__}")

import pymeshlab
check("PyMeshLab", lambda: f"v{pymeshlab.__version__}")

import cv2
check("OpenCV", lambda: f"v{cv2.__version__}")

print("\n--- 3. Eigene C++/CUDA Extensions (Kritisch!) ---")
import custom_rasterizer
check("Custom Rasterizer (C++/CUDA .so)", lambda: "Erfolgreich geladen")

sys.path.append("/opt/hunyuan3d/hy3dpaint/DifferentiableRenderer")
import mesh_inpaint_processor
check("Mesh Inpaint Processor (C++ .so)", lambda: "Erfolgreich geladen")

print("\n--- 4. Hunyuan3D Codebase & Pipeline-Klassen ---")
import hy3dgen
check("Hunyuan3D Core (hy3dgen)", lambda: "Erfolgreich geladen")

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
check("ShapeGen Pipeline Klasse", lambda: Hunyuan3DDiTFlowMatchingPipeline.__name__)

from hy3dgen.texturing import Hunyuan3DPaintPipeline
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
if failed_tests == 0:
    print(f"{GREEN}🎉 ALLE PYTHON-TESTS BESTANDEN!{RESET}")
    sys.exit(0)
else:
    print(f"{RED}❌ {failed_tests} TEST(S) FEHLGESCHLAGEN!{RESET}")
    sys.exit(1)
EOF

PYTHON_EXIT_CODE=$?

echo ""
echo "--- 6. CLI-Skript Parser Test (main.py) ---"
docker run --rm "$IMAGE_NAME" python3 /opt/hunyuan3d/main.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "[\033[92mOK\033[0m] /opt/hunyuan3d/main.py --help reagiert fehlerfrei"
else
    echo -e "[\033[91mFAIL\033[0m] /opt/hunyuan3d/main.py --help wirft Fehler"
    PYTHON_EXIT_CODE=1
fi

echo "============================================================"
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo -e "\033[92mBereit für RunPod! Das Demoskript wird mit einer NVIDIA-GPU laufen.\033[0m"
else
    echo -e "\033[91mBitte prüfe die Fehlermeldungen oben im Protokoll.\033[0m"
fi
