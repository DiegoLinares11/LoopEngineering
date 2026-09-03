"""Calcula las atribuciones SHAP de ViT-B/16 sobre las tres imagenes de futbol.

Usa el Partition explainer con mascara de imagen (`inpaint_telea`), que es
agnostico al modelo: solo necesita la funcion de prediccion, no los gradientes
internos del transformer. A cambio es una aproximacion muestreada de los valores
de Shapley, asi que el presupuesto de evaluaciones (--evals) es un parametro del
experimento y queda registrado en la salida.

Guarda los valores crudos en un .npz para que el analisis cuantitativo no tenga
que volver a pagar el costo de computo.
"""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from common import (
    FIGURES_DIR,
    RESULTS_DIR,
    cargar_imagenes,
    cargar_modelo,
    fijar_semillas,
    hacer_funcion_prediccion,
    indices_de_interes,
    versiones,
)

CLASE_OBJETIVO = "soccer ball"
TITULOS = {
    "close_up": "A. Close-up (contexto minimo)",
    "campo": "B. Balon sobre el campo",
    "partido": "C. Escena de partido (contexto maximo)",
}


def mapa_de_atribucion(valores: np.ndarray) -> np.ndarray:
    """Colapsa los canales RGB en un solo mapa de atribucion por pixel."""
    return valores.sum(axis=-1)


def figura(ids, imagenes, mapas, probs, destino):
    n = len(ids)
    fig, ejes = plt.subplots(n, 2, figsize=(7.2, 3.4 * n))
    for fila, (nombre, img, mapa, prob) in enumerate(zip(ids, imagenes, mapas, probs)):
        # Se muestra en milesimas de logit: los valores por pixel son diminutos y
        # una escala en notacion cientifica es ilegible en la figura impresa.
        mapa = mapa * 1e3
        limite = np.percentile(np.abs(mapa), 99.5) or np.abs(mapa).max()

        ax = ejes[fila, 0]
        ax.imshow(img.astype(np.uint8))
        ax.set_title(TITULOS.get(nombre, nombre), fontsize=9, loc="left")
        ax.set_ylabel(f"p({CLASE_OBJETIVO}) = {prob:.3f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = ejes[fila, 1]
        gris = img.astype(np.float32).mean(axis=-1) / 255.0
        ax.imshow(gris, cmap="gray", vmin=0, vmax=1)
        im = ax.imshow(mapa, cmap="bwr", vmin=-limite, vmax=limite, alpha=0.75)
        ax.set_title("Atribucion SHAP", fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        barra = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        barra.ax.tick_params(labelsize=6)
        barra.set_label(r"contribucion al logit por pixel ($	imes 10^{-3}$)", fontsize=6)

    fig.suptitle(
        f"Evidencia a favor de la clase '{CLASE_OBJETIVO}' (rojo suma, azul resta)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destino, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", type=int, default=2000, help="presupuesto de evaluaciones por imagen")
    parser.add_argument("--batch", type=int, default=50, help="imagenes enmascaradas por lote")
    args = parser.parse_args()

    fijar_semillas()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    processor, model, id2label = cargar_modelo()
    objetivo = indices_de_interes(id2label)[CLASE_OBJETIVO]
    f = hacer_funcion_prediccion(processor, model)

    ids, imagenes = cargar_imagenes()
    X = imagenes.astype(np.float32)
    probs = torch.softmax(torch.from_numpy(f(X)), dim=-1).numpy()[:, objetivo]

    masker = shap.maskers.Image("inpaint_telea", X[0].shape)
    explainer = shap.Explainer(f, masker, output_names=[id2label[i] for i in range(len(id2label))])

    mapas, tiempos = [], []
    for i, nombre in enumerate(ids):
        # Se mide tiempo de CPU y no de reloj: si la maquina se suspende a media
        # corrida el reloj de pared cuenta la suspension y el dato deja de servir.
        inicio = time.process_time()
        sv = explainer(
            X[i : i + 1],
            max_evals=args.evals,
            batch_size=args.batch,
            outputs=[objetivo],
        )
        tiempos.append(time.process_time() - inicio)
        mapas.append(mapa_de_atribucion(sv.values[0, ..., 0]))
        print(f"[{nombre}] {tiempos[-1]:.1f}s CPU  suma={mapas[-1].sum():+.4f}", flush=True)

    mapas = np.stack(mapas)
    np.savez_compressed(
        RESULTS_DIR / "shap_values.npz",
        ids=np.array(ids),
        imagenes=imagenes,
        mapas=mapas,
        prob_objetivo=probs,
        indice_objetivo=objetivo,
        evals=args.evals,
    )
    figura(ids, imagenes, mapas, probs, FIGURES_DIR / "shap_atribuciones.png")

    resumen = {
        "clase_objetivo": CLASE_OBJETIVO,
        "indice_objetivo": objetivo,
        "evaluaciones_por_imagen": args.evals,
        "versiones": versiones(),
        "imagenes": [
            {
                "id": nombre,
                "prob_objetivo": float(p),
                "suma_atribuciones": float(m.sum()),
                "masa_positiva": float(m[m > 0].sum()),
                "masa_negativa": float(m[m < 0].sum()),
                "segundos_cpu": round(t, 1),
            }
            for nombre, p, m, t in zip(ids, probs, mapas, tiempos)
        ],
    }
    (RESULTS_DIR / "shap_summary.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nFigura  -> {FIGURES_DIR / 'shap_atribuciones.png'}")
    print(f"Valores -> {RESULTS_DIR / 'shap_values.npz'}")


if __name__ == "__main__":
    main()
