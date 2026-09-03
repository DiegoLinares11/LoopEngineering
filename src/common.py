"""Piezas compartidas por los scripts: rutas, carga del modelo y preprocesamiento.

Todo lo que afecte a los numeros del paper vive aqui, para que clasificacion y
explicacion vean exactamente el mismo modelo y los mismos pixeles.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "google/vit-base-patch16-224"
INPUT_SIZE = 224
SEED = 20260902

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Clases de ImageNet-1k que compiten en una escena de futbol. Se resuelven por
# nombre contra el id2label del modelo para no depender de indices escritos a mano.
CLASES_DE_INTERES = [
    "soccer ball",
    "ballplayer, baseball player",
    "rugby ball",
    "basketball",
    "volleyball",
    "scoreboard",
]


def fijar_semillas(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def cargar_modelo():
    """Devuelve (processor, model, id2label) con el modelo en modo evaluacion."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model, model.config.id2label


def indices_de_interes(id2label: dict[int, str]) -> dict[str, int]:
    """Mapea el nombre de cada clase de interes a su indice en ImageNet-1k."""
    por_nombre = {nombre: idx for idx, nombre in id2label.items()}
    faltantes = [c for c in CLASES_DE_INTERES if c not in por_nombre]
    if faltantes:
        raise KeyError(f"clases ausentes en el modelo: {faltantes}")
    return {c: por_nombre[c] for c in CLASES_DE_INTERES}


def recortar_centro(img: Image.Image, size: int = INPUT_SIZE) -> Image.Image:
    """Escala el lado corto a `size` y recorta el centro, sin deformar la escena."""
    w, h = img.size
    escala = size / min(w, h)
    img = img.resize((round(w * escala), round(h * escala)), Image.BICUBIC)
    w, h = img.size
    izq, arriba = (w - size) // 2, (h - size) // 2
    return img.crop((izq, arriba, izq + size, arriba + size))


def cargar_imagenes() -> tuple[list[str], np.ndarray]:
    """Carga las imagenes del manifiesto ya recortadas: (ids, array uint8 NxHxWx3)."""
    manifest = json.loads((IMAGES_DIR / "manifest.json").read_text(encoding="utf-8"))
    ids, arrays = [], []
    for item in manifest["imagenes"]:
        img = Image.open(IMAGES_DIR / item["archivo"]).convert("RGB")
        ids.append(item["id"])
        arrays.append(np.array(recortar_centro(img)))
    return ids, np.stack(arrays)


def hacer_funcion_prediccion(processor, model):
    """Envuelve el modelo en f(batch uint8 NxHxWx3) -> logits NxC, como espera SHAP."""

    def f(x: np.ndarray) -> np.ndarray:
        imgs = [Image.fromarray(a.astype(np.uint8)) for a in x]
        entradas = processor(images=imgs, return_tensors="pt")
        with torch.no_grad():
            return model(**entradas).logits.numpy()

    return f


def versiones() -> dict[str, str]:
    """Versiones que hay que citar para que una corrida sea rastreable."""
    import shap
    import transformers

    return {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "shap": shap.__version__,
        "numpy": np.__version__,
        "modelo": MODEL_ID,
        "semilla": str(SEED),
    }
