#!/usr/bin/env python3
"""Valida la integridad del paquete antes de entregarlo, sin usar la red."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ERRORES: list[str] = []
COMPROBACIONES = 0


def comprobar(condicion: bool, mensaje: str) -> None:
    global COMPROBACIONES
    COMPROBACIONES += 1
    if not condicion:
        ERRORES.append(mensaje)


def cargar_json(ruta: Path) -> Any | None:
    comprobar(ruta.is_file(), f"Falta {ruta.relative_to(ROOT)}")
    if not ruta.is_file():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        ERRORES.append(f"JSON inválido en {ruta.relative_to(ROOT)}: {exc}")
        return None


OBLIGATORIOS = [
    "README.md",
    "requirements.txt",
    "analisis_shap.ipynb",
    "paper/paper.tex",
    "paper/paper.pdf",
    "paper/refs.bib",
    "docs/prompt_log.md",
    "docs/limitations.md",
    "results/predictions.json",
    "results/evaluation.json",
    "results/shap_values.npz",
]
for nombre in OBLIGATORIOS:
    ruta = ROOT / nombre
    comprobar(ruta.is_file() and ruta.stat().st_size > 0, f"Falta o está vacío: {nombre}")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
comprobar("## La pregunta" in readme and "?" in readme, "README sin pregunta explicativa")
comprobar("google/vit-base-patch16-224" in readme, "README sin clasificador preentrenado")

notebook = cargar_json(ROOT / "analisis_shap.ipynb")
if isinstance(notebook, dict):
    comprobar(isinstance(notebook.get("nbformat"), int), "Notebook sin versión nbformat")
    comprobar(bool(notebook.get("cells")), "Notebook sin celdas")
    contenido_notebook = json.dumps(notebook, ensure_ascii=False)
    comprobar("SHAP" in contenido_notebook, "Notebook sin análisis SHAP")
    comprobar("Andy Fuentes" in contenido_notebook, "Notebook sin Andy Fuentes como autor")

for nombre in ("predictions.json", "evaluation.json"):
    resultado = cargar_json(ROOT / "results" / nombre)
    if isinstance(resultado, dict):
        comprobar(bool(resultado.get("versiones")), f"{nombre} no registra versiones")

manifiesto = cargar_json(ROOT / "data/images/manifest.json")
anotaciones = cargar_json(ROOT / "data/annotations.json")
ids: set[str] = set()
if isinstance(manifiesto, dict):
    imagenes = manifiesto.get("imagenes", [])
    comprobar(isinstance(imagenes, list) and 1 <= len(imagenes) <= 3, "Se esperaban de 1 a 3 imágenes")
    if isinstance(imagenes, list):
        for entrada in imagenes:
            if not isinstance(entrada, dict):
                ERRORES.append("Entrada inválida en el manifiesto de imágenes")
                continue
            id_imagen = str(entrada.get("id", ""))
            ids.add(id_imagen)
            for campo in ("id", "archivo", "titulo_commons", "pagina", "autor", "licencia", "sha256"):
                comprobar(bool(entrada.get(campo)), f"Falta {campo} en la imagen {id_imagen or '(sin id)'}")
            ruta_imagen = ROOT / "data/images" / str(entrada.get("archivo", ""))
            comprobar(ruta_imagen.is_file(), f"No existe la imagen declarada: {ruta_imagen.name}")
            if ruta_imagen.is_file():
                hash_real = hashlib.sha256(ruta_imagen.read_bytes()).hexdigest()
                comprobar(hash_real == entrada.get("sha256"), f"Hash incorrecto para {ruta_imagen.name}")

if isinstance(anotaciones, dict):
    cajas = anotaciones.get("cajas", {})
    comprobar(isinstance(cajas, dict) and set(cajas) == ids, "Las cajas no coinciden con las imágenes")
    if isinstance(cajas, dict):
        for id_imagen, caja in cajas.items():
            valida = isinstance(caja, list) and len(caja) == 4 and all(isinstance(x, int) for x in caja)
            comprobar(valida, f"Caja inválida para {id_imagen}")

tex_path = ROOT / "paper/paper.tex"
tex = tex_path.read_text(encoding="utf-8")
comprobar("Andy Fuentes" in tex, "paper.tex sin Andy Fuentes como autor")

for patron, extension in ((r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", ""), (r"\\input\{([^}]+)\}", ".tex")):
    for referencia in re.findall(patron, tex):
        destino = tex_path.parent / referencia
        if extension and not destino.suffix:
            destino = destino.with_suffix(extension)
        comprobar(destino.is_file(), f"Recurso LaTeX ausente: {referencia}")

for referencia in re.findall(r"\\bibliography\{([^}]+)\}", tex):
    for nombre in referencia.split(","):
        destino = tex_path.parent / f"{nombre.strip()}.bib"
        comprobar(destino.is_file(), f"Bibliografía ausente: {nombre.strip()}.bib")

pdf = ROOT / "paper/paper.pdf"
comprobar(pdf.read_bytes()[:5] == b"%PDF-", "paper.pdf no es un PDF válido")

if ERRORES:
    print(f"FALLO: {len(ERRORES)} problema(s) en {COMPROBACIONES} comprobaciones:")
    for error in ERRORES:
        print(f"- {error}")
    sys.exit(1)

print(f"OK: paquete completo; {COMPROBACIONES} comprobaciones superadas.")
