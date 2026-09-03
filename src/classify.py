"""Clasifica las tres imagenes con ViT-B/16 y guarda las predicciones en JSON.

Es el paso previo a SHAP: define cual es la clase que hay que explicar en cada
imagen y deja por escrito con cuanta confianza la predijo el modelo.
"""

import json

import numpy as np
import torch

from common import (
    RESULTS_DIR,
    cargar_imagenes,
    cargar_modelo,
    fijar_semillas,
    hacer_funcion_prediccion,
    indices_de_interes,
    versiones,
)

TOP_K = 5


def main() -> None:
    fijar_semillas()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    processor, model, id2label = cargar_modelo()
    interes = indices_de_interes(id2label)
    f = hacer_funcion_prediccion(processor, model)

    ids, imagenes = cargar_imagenes()
    probs = torch.softmax(torch.from_numpy(f(imagenes)), dim=-1).numpy()

    salida = []
    for nombre, p in zip(ids, probs):
        orden = np.argsort(p)[::-1][:TOP_K]
        salida.append(
            {
                "id": nombre,
                "clase_predicha": id2label[int(orden[0])],
                "indice_predicho": int(orden[0]),
                "confianza": float(p[orden[0]]),
                "top_k": [
                    {"clase": id2label[int(i)], "indice": int(i), "prob": float(p[i])}
                    for i in orden
                ],
                "clases_de_interes": {
                    clase: {"indice": idx, "prob": float(p[idx])}
                    for clase, idx in interes.items()
                },
            }
        )
        print(f"\n[{nombre}] -> {salida[-1]['clase_predicha']} ({salida[-1]['confianza']:.3f})")
        for entrada in salida[-1]["top_k"]:
            print(f"    {entrada['prob']:.4f}  {entrada['clase']}")

    destino = RESULTS_DIR / "predictions.json"
    destino.write_text(
        json.dumps({"versiones": versiones(), "predicciones": salida}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nPredicciones -> {destino}")


if __name__ == "__main__":
    main()
