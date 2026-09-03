"""Somete las atribuciones SHAP a pruebas que pueden refutarlas.

Un mapa de calor bonito no demuestra nada: hay que comprobar que los pixeles que
SHAP senala son de verdad los que sostienen la prediccion. Se calculan tres cosas:

1. Concentracion: que fraccion de la evidencia positiva cae dentro del balon,
   comparada con la que caeria ahi por puro azar (enriquecimiento).
2. Curva de supresion: se borran los pixeles en orden de atribucion decreciente y
   se mide como cae p(soccer ball). Si SHAP acierta, esa caida debe ser mas
   rapida que borrando pixeles al azar.
3. Ablaciones: dejar solo el balon, o borrar solo el balon. Es la prueba causal
   directa de si el objeto basta y si es necesario.

El contenido "borrado" se sustituye por una version muy desenfocada de la propia
imagen, que destruye la forma local sin introducir bordes artificiales.
"""

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    FIGURES_DIR,
    RESULTS_DIR,
    ROOT,
    cargar_modelo,
    fijar_semillas,
    hacer_funcion_prediccion,
    versiones,
)

FRACCIONES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
SEMILLAS_AZAR = [0, 1, 2]
SIGMA_DESENFOQUE = 15
TITULOS = {"close_up": "A. Close-up", "campo": "B. Balon en el campo", "partido": "C. Escena de partido"}


def desenfocar(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img.astype(np.uint8), (0, 0), SIGMA_DESENFOQUE).astype(np.float32)


def componer(img: np.ndarray, fondo: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Devuelve la imagen con los pixeles marcados sustituidos por el fondo."""
    salida = img.copy()
    salida[mascara] = fondo[mascara]
    return salida


def prob_objetivo(f, lote: np.ndarray, indice: int) -> np.ndarray:
    logits = torch.from_numpy(f(np.stack(lote)))
    return torch.softmax(logits, dim=-1).numpy()[:, indice]


def curva_supresion(f, img, fondo, mapa, indice, orden_azar=None):
    """p(clase) al ir borrando fracciones crecientes de pixeles."""
    n = mapa.size
    if orden_azar is None:
        orden = np.argsort(mapa, axis=None)[::-1]  # de mayor a menor atribucion
    else:
        orden = orden_azar
    lote = []
    for q in FRACCIONES:
        mascara = np.zeros(n, dtype=bool)
        mascara[orden[: int(round(q * n))]] = True
        lote.append(componer(img, fondo, mascara.reshape(mapa.shape)[..., None].repeat(3, axis=-1)))
    return prob_objetivo(f, lote, indice)


def main() -> None:
    fijar_semillas()
    datos = np.load(RESULTS_DIR / "shap_values.npz", allow_pickle=False)
    ids = [str(x) for x in datos["ids"]]
    imagenes = datos["imagenes"].astype(np.float32)
    mapas = datos["mapas"]
    indice = int(datos["indice_objetivo"])
    cajas = json.loads((ROOT / "data" / "annotations.json").read_text(encoding="utf-8"))["cajas"]

    processor, model, _ = cargar_modelo()
    f = hacer_funcion_prediccion(processor, model)

    filas, curvas = [], {}
    for nombre, img, mapa in zip(ids, imagenes, mapas):
        fondo = desenfocar(img)
        x0, y0, x1, y1 = cajas[nombre]
        dentro = np.zeros(mapa.shape, dtype=bool)
        dentro[y0:y1, x0:x1] = True

        # 1. concentracion de la evidencia positiva
        positivo = np.clip(mapa, 0, None)
        frac_masa = float(positivo[dentro].sum() / positivo.sum())
        frac_area = float(dentro.mean())
        enriquecimiento = frac_masa / frac_area

        # 2. curvas de supresion
        guiada = curva_supresion(f, img, fondo, mapa, indice)
        azar = np.mean(
            [
                curva_supresion(
                    f, img, fondo, mapa, indice,
                    orden_azar=np.random.default_rng(s).permutation(mapa.size),
                )
                for s in SEMILLAS_AZAR
            ],
            axis=0,
        )
        curvas[nombre] = {"guiada": guiada.tolist(), "azar": azar.tolist()}

        # 3. ablaciones causales
        mascara3 = dentro[..., None].repeat(3, axis=-1)
        solo_objeto, sin_objeto = prob_objetivo(
            f, [componer(img, fondo, ~mascara3), componer(img, fondo, mascara3)], indice
        )

        filas.append(
            {
                "id": nombre,
                "prob_original": float(guiada[0]),
                "area_objeto": round(frac_area, 4),
                "masa_positiva_en_objeto": round(frac_masa, 4),
                "enriquecimiento": round(enriquecimiento, 2),
                "auc_supresion_guiada": round(float(np.trapezoid(guiada, FRACCIONES)), 4),
                "auc_supresion_azar": round(float(np.trapezoid(azar, FRACCIONES)), 4),
                "prob_solo_objeto": float(solo_objeto),
                "prob_sin_objeto": float(sin_objeto),
            }
        )
        r = filas[-1]
        print(
            f"[{nombre}] p={r['prob_original']:.3f} | masa en balon={frac_masa:.1%} "
            f"(area {frac_area:.1%}, x{enriquecimiento:.1f}) | "
            f"AUC guiada={r['auc_supresion_guiada']:.3f} vs azar={r['auc_supresion_azar']:.3f} | "
            f"solo balon={solo_objeto:.3f} sin balon={sin_objeto:.3f}",
            flush=True,
        )

    figura_curvas(ids, curvas, FIGURES_DIR / "curvas_supresion.png")
    (RESULTS_DIR / "evaluation.json").write_text(
        json.dumps(
            {
                "fracciones": FRACCIONES,
                "sigma_desenfoque": SIGMA_DESENFOQUE,
                "semillas_azar": SEMILLAS_AZAR,
                "versiones": versiones(),
                "metricas": filas,
                "curvas": curvas,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nMetricas -> {RESULTS_DIR / 'evaluation.json'}")


def figura_curvas(ids, curvas, destino):
    fig, ejes = plt.subplots(1, len(ids), figsize=(3.4 * len(ids), 3.2), sharey=True)
    for ax, nombre in zip(np.atleast_1d(ejes), ids):
        ax.plot(FRACCIONES, curvas[nombre]["guiada"], "o-", color="crimson", lw=1.8,
                ms=3.5, label="orden SHAP")
        ax.plot(FRACCIONES, curvas[nombre]["azar"], "s--", color="steelblue", lw=1.4,
                ms=3, label="orden aleatorio")
        ax.set_title(TITULOS.get(nombre, nombre), fontsize=9)
        ax.set_xlabel("fraccion de pixeles borrados", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, lw=0.5)
    np.atleast_1d(ejes)[0].set_ylabel("p(soccer ball)", fontsize=8)
    np.atleast_1d(ejes)[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura   -> {destino}")


if __name__ == "__main__":
    main()
