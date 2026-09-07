#!/usr/bin/env python3
"""Batch-Texturierung von .obj Shapes mit Referenzbildern.

Exportiert fertige .glb Modelle. Optimiert für das RunPod-Container-Template.
"""

import argparse
import gc
import os
from pathlib import Path
import sys
from PIL import Image
import torch
from tqdm import tqdm

# Headless EGL-Rendering für Container erzwingen
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

HUNYUAN_ROOT = Path(os.environ.get("HUNYUAN_ROOT", "/opt/hunyuan3d")).resolve()
if not HUNYUAN_ROOT.exists():
  HUNYUAN_ROOT = (
      Path(__file__).resolve().parent
      if (Path(__file__).resolve().parent / "hy3dpaint").exists()
      else Path.cwd()
  )

sys.path.insert(0, str(HUNYUAN_ROOT))
sys.path.insert(0, str(HUNYUAN_ROOT / "hy3dshape"))
sys.path.insert(0, str(HUNYUAN_ROOT / "hy3dpaint"))

# Torchvision-Fix
try:
  from torchvision_fix import apply_fix

  apply_fix()
except ImportError:
  pass
except Exception as e:
  print(f"Hinweis zum Torchvision-Fix: {e}")

from hy3dshape.rembg import BackgroundRemover
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".PNG",
    ".JPG",
    ".JPEG",
}


def find_matching_image(mesh_stem: str, search_dirs: list[Path]) -> Path | None:
  for directory in search_dirs:
    if not directory or not directory.is_dir():
      continue
    for ext in IMAGE_EXTENSIONS:
      candidate = directory / f"{mesh_stem}{ext}"
      if candidate.exists():
        return candidate
  return None


def preprocess_image(image_path: Path, rembg_worker: BackgroundRemover = None):
  image = Image.open(image_path)
  has_alpha = False
  if image.mode in ("RGBA", "LA") or (
      image.mode == "P" and "transparency" in image.info
  ):
    alpha = image.convert("RGBA").split()[-1]
    if alpha.getextrema()[0] < 250:
      has_alpha = True

  if not has_alpha and rembg_worker is not None:
    image = rembg_worker(image.convert("RGB"))
  else:
    image = image.convert("RGBA")
  return image


def main():
  parser = argparse.ArgumentParser(
      description="RunPod Batch 3D-Texturierung (OBJ -> GLB)"
  )
  parser.add_argument(
      "--input_dir",
      "-i",
      type=str,
      required=True,
      help="Pfad zum Ordner mit untexturierten .obj Dateien",
  )
  parser.add_argument(
      "--output_dir",
      "-o",
      type=str,
      default="/workspace/outputs_textured",
      help="Pfad zum Ausgabeordner",
  )
  parser.add_argument(
      "--image_dir",
      type=str,
      default=None,
      help="Pfad zum Bildordner (Standard: sucht in --image_dir, dann in --input_dir)",
  )
  parser.add_argument(
      "--fallback_image",
      type=str,
      default=None,
      help="Standardbild falls kein passender Name gefunden wird",
  )
  parser.add_argument(
      "--max_num_view",
      type=int,
      default=6,
      choices=range(6, 10),
      help="Blickwinkel (6-9)",
  )
  parser.add_argument(
      "--resolution",
      type=int,
      default=512,
      choices=[512, 768],
      help="Textur-Auflösung (512 oder 768)",
  )
  parser.add_argument(
      "--no_rembg",
      action="store_true",
      help="Automatische Hintergrundentfernung deaktivieren",
  )
  parser.add_argument(
      "--skip_existing",
      action="store_true",
      help="Existierende .glb Modelle überspringen",
  )

  args = parser.parse_args()

  input_dir = Path(args.input_dir).resolve()
  output_dir = Path(args.output_dir).resolve()
  image_dir = Path(args.image_dir).resolve() if args.image_dir else None
  fallback_image_path = (
      Path(args.fallback_image).resolve() if args.fallback_image else None
  )

  if not input_dir.is_dir():
    print(
        f"FEHLER: Eingabeverzeichnis nicht gefunden: {input_dir}",
        file=sys.stderr,
    )
    sys.exit(1)

  output_dir.mkdir(parents=True, exist_ok=True)
  obj_files = sorted(
      [f for f in input_dir.iterdir() if f.suffix.lower() == ".obj"]
  )

  if not obj_files:
    print(f"Keine .obj-Dateien in {input_dir} gefunden.")
    sys.exit(0)

  print(f"[RunPod] Initialisiere Texturierungs-Pipeline...")

  # Absolute Pfade zu den vorinstallierten Checkpoints und Configs in /opt/hunyuan3d
  conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
  conf.realesrgan_ckpt_path = str(
      HUNYUAN_ROOT / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
  )
  conf.multiview_cfg_path = str(
      HUNYUAN_ROOT / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
  )
  conf.custom_pipeline = str(HUNYUAN_ROOT / "hy3dpaint/hunyuanpaintpbr")

  paint_pipeline = Hunyuan3DPaintPipeline(conf)

  rembg_worker = None
  if not args.no_rembg:
    print("[RunPod] Initialisiere rembg...")
    rembg_worker = BackgroundRemover()

  search_dirs = [image_dir, input_dir] if image_dir else [input_dir]
  successful, skipped, failed = 0, 0, 0

  print(f"[RunPod] Starte Texturierung von {len(obj_files)} Modellen...")

  for obj_path in tqdm(obj_files, desc="Texturiere Meshes"):
    target_glb = output_dir / f"{obj_path.stem}.glb"
    if args.skip_existing and target_glb.exists():
      skipped += 1
      continue

    ref_image_path = find_matching_image(obj_path.stem, search_dirs)
    if ref_image_path is None:
      if fallback_image_path and fallback_image_path.exists():
        ref_image_path = fallback_image_path
      else:
        print(
            f'\n[WARNUNG] Kein Referenzbild für "{obj_path.name}" gefunden.'
            " Überspringe..."
        )
        failed += 1
        continue

    try:
      ref_image = preprocess_image(ref_image_path, rembg_worker=rembg_worker)

      with torch.inference_mode():
        paint_pipeline(
            mesh_path=str(obj_path),
            image_path=ref_image,
            output_mesh_path=str(target_glb),
        )
      successful += 1

    except Exception as e:
      print(f'\n[FEHLER] Texturierung fehlgeschlagen für "{obj_path.name}": {e}')
      failed += 1

    finally:
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      gc.collect()

  print("\n" + "=" * 45)
  print("Batch-Texturierung beendet:")
  print(f"Erfolgreich:   {successful}")
  print(f"Übersprungen:  {skipped}")
  print(f"Fehler/Fehlend:{failed}")
  print(f"Ausgabeordner: {output_dir}")
  print("=" * 45)


if __name__ == "__main__":
  main()