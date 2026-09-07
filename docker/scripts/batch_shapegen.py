#!/usr/bin/env python3
"""Batch-Generierung von 3D-Modellen (ohne Textur) aus Bildern.

Optimiert für das RunPod-Container-Template.
"""

import argparse
import gc
import os
from pathlib import Path
import sys
from PIL import Image
import torch
from tqdm import tqdm

# Absoluten Pfad zum Repository ermitteln (/opt/hunyuan3d)
HUNYUAN_ROOT = Path(os.environ.get("HUNYUAN_ROOT", "/opt/hunyuan3d")).resolve()
if not HUNYUAN_ROOT.exists():
  # Fallback: Ordner der Datei oder aktuelles Verzeichnis
  HUNYUAN_ROOT = (
      Path(__file__).resolve().parent
      if (Path(__file__).resolve().parent / "hy3dshape").exists()
      else Path.cwd()
  )

sys.path.insert(0, str(HUNYUAN_ROOT))
sys.path.insert(0, str(HUNYUAN_ROOT / "hy3dshape"))

# Torchvision-Fix
try:
  from torchvision_fix import apply_fix

  apply_fix()
except ImportError:
  pass
except Exception as e:
  print(f"Hinweis zum Torchvision-Fix: {e}")

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".PNG",
    ".JPG",
    ".JPEG",
}


def has_transparency(image: Image.Image) -> bool:
  if image.mode in ("RGBA", "LA") or (
      image.mode == "P" and "transparency" in image.info
  ):
    alpha = image.convert("RGBA").split()[-1]
    return alpha.getextrema()[0] < 250
  return False


def preprocess_image(
    image_path: Path, rembg_worker: BackgroundRemover, force_rembg: bool = False
) -> Image.Image:
  image = Image.open(image_path)
  if has_transparency(image) and not force_rembg:
    return image.convert("RGBA")

  if rembg_worker is not None:
    image = rembg_worker(image.convert("RGB"))
  else:
    image = image.convert("RGBA")
  return image


def main():
  parser = argparse.ArgumentParser(
      description="RunPod Batch 3D-Shape-Generierung (ohne Textur)"
  )
  parser.add_argument(
      "--input_dir",
      "-i",
      type=str,
      required=True,
      help="Pfad zum Eingabeordner mit Bildern (z. B. /workspace/inputs)",
  )
  parser.add_argument(
      "--output_dir",
      "-o",
      type=str,
      default="/workspace/outputs_shapes",
      help="Pfad zum Ausgabeordner",
  )
  parser.add_argument(
      "--model_path",
      type=str,
      default="tencent/Hunyuan3D-2.1",
      help="HuggingFace Modell-ID oder lokaler Cache-Pfad",
  )
  parser.add_argument(
      "--format",
      type=str,
      default="glb",
      choices=["glb", "obj"],
      help="Dateiformat (glb oder obj)",
  )
  parser.add_argument(
      "--octree_resolution",
      type=int,
      default=256,
      help="Mesh-Auflösung (z. B. 256 oder 384)",
  )
  parser.add_argument(
      "--num_inference_steps",
      type=int,
      default=None,
      help="Diffusionsschritte",
  )
  parser.add_argument(
      "--guidance_scale", type=float, default=None, help="Guidance Scale"
  )
  parser.add_argument(
      "--seed", type=int, default=None, help="Reproduzierbarer Seed"
  )
  parser.add_argument(
      "--no_rembg",
      action="store_true",
      help="Hintergrundentfernung überspringen",
  )
  parser.add_argument(
      "--force_rembg",
      action="store_true",
      help="Hintergrundentfernung immer erzwingen",
  )
  parser.add_argument(
      "--skip_existing",
      action="store_true",
      help="Bereits existierende Modelle überspringen",
  )
  parser.add_argument(
      "--device",
      type=str,
      default="cuda" if torch.cuda.is_available() else "cpu",
  )

  args = parser.parse_args()

  input_dir = Path(args.input_dir).resolve()
  output_dir = Path(args.output_dir).resolve()

  if not input_dir.is_dir():
    print(
        f"FEHLER: Eingabeverzeichnis nicht gefunden: {input_dir}",
        file=sys.stderr,
    )
    sys.exit(1)

  output_dir.mkdir(parents=True, exist_ok=True)
  image_files = sorted(
      [f for f in input_dir.iterdir() if f.suffix in SUPPORTED_EXTENSIONS]
  )

  if not image_files:
    print(f"Keine Bilder in {input_dir} gefunden.")
    sys.exit(0)

  print(f"[RunPod] Verarbeite {len(image_files)} Bilder aus {input_dir}")
  print(f"[RunPod] Lade Modell: {args.model_path}")

  pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
      args.model_path
  )
  if hasattr(pipeline_shapegen, "to"):
    pipeline_shapegen.to(args.device)

  rembg_worker = None
  if not args.no_rembg:
    print("[RunPod] Initialisiere rembg...")
    rembg_worker = BackgroundRemover()

  generator = None
  if args.seed is not None:
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

  successful, skipped, failed = 0, 0, 0

  for img_path in tqdm(image_files, desc="Generiere 3D-Shapes"):
    target_file = output_dir / f"{img_path.stem}.{args.format}"
    if args.skip_existing and target_file.exists():
      skipped += 1
      continue

    try:
      img = preprocess_image(
          img_path, rembg_worker, force_rembg=args.force_rembg
      )

      pipe_kwargs = {
          "image": img,
          "octree_resolution": args.octree_resolution,
      }
      if args.num_inference_steps is not None:
        pipe_kwargs["num_inference_steps"] = args.num_inference_steps
      if args.guidance_scale is not None:
        pipe_kwargs["guidance_scale"] = args.guidance_scale
      if generator is not None:
        pipe_kwargs["generator"] = generator

      with torch.inference_mode():
        outputs = pipeline_shapegen(**pipe_kwargs)
        mesh = outputs[0]

      mesh.export(str(target_file))
      successful += 1

    except Exception as e:
      print(f'\n[FEHLER] Bild "{img_path.name}": {e}')
      failed += 1

    finally:
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      gc.collect()

  print("\n" + "=" * 45)
  print("Batch Shape-Generierung beendet:")
  print(f"Erfolgreich:   {successful}")
  print(f"Übersprungen:  {skipped}")
  print(f"Fehler:        {failed}")
  print(f"Ausgabeordner: {output_dir}")
  print("=" * 45)


if __name__ == "__main__":
  main()