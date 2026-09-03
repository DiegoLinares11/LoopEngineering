"""Genera las tablas del paper a partir de los JSON de resultados.

Ninguna cifra del documento se escribe a mano: si cambia el experimento, cambian
las tablas al volver a ejecutar este script. Es la garantia de que el texto no
puede desincronizarse de lo que realmente calculo el codigo.

Las plantillas usan cadenas crudas (rf) porque en una cadena normal de Python las
secuencias de LaTeX se interpretan como caracteres de control: la "b" de \\begin y
la "t" de \\toprule desaparecerian sin ningun error visible.
"""

import json

from common import RESULTS_DIR, ROOT

SALIDA = ROOT / "paper" / "generado"
ETIQUETAS = {
    "close_up": "A. Close-up",
    "campo": "B. Balon en el campo",
    "partido": "C. Escena de partido",
}
AVISO = "% Generado por src/make_tables.py. No editar a mano."
FIN = r" \\"


def tabla_predicciones(pred: dict) -> str:
    filas = []
    for p in pred["predicciones"]:
        segunda = p["top_k"][1]
        filas.append(
            rf"{ETIQUETAS[p['id']]} & \texttt{{{p['clase_predicha']}}} & {p['confianza']:.3f} & "
            rf"\texttt{{{segunda['clase']}}} & {segunda['prob']:.3f}" + FIN
        )
    cuerpo = "\n".join(filas)
    return rf"""{AVISO}
\begin{{tabular}}{{llrlr}}
\toprule
Imagen & Clase predicha & $p$ & Segunda clase & $p$""" + FIN + rf"""
\midrule
{cuerpo}
\bottomrule
\end{{tabular}}
"""


def tabla_metricas(ev: dict) -> str:
    filas = []
    for m in ev["metricas"]:
        filas.append(
            rf"{ETIQUETAS[m['id']]} & {m['area_objeto']*100:.1f} & "
            rf"{m['masa_positiva_en_objeto']*100:.1f} & {m['enriquecimiento']:.1f} & "
            rf"{m['auc_supresion_guiada']:.3f} & {m['auc_supresion_azar']:.3f} & "
            rf"{m['prob_solo_objeto']:.3f} & {m['prob_sin_objeto']:.3f}" + FIN
        )
    cuerpo = "\n".join(filas)
    return (
        rf"""{AVISO}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
& \multicolumn{{3}}{{c}}{{Concentracion}} & \multicolumn{{2}}{{c}}{{AUC supresion}}"""
        rf""" & \multicolumn{{2}}{{c}}{{Ablacion}}""" + FIN + rf"""
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-6}} \cmidrule(lr){{7-8}}
Imagen & Area & Masa & Enriq. & SHAP & Azar & Solo balon & Sin balon""" + FIN + rf"""
& (\%) & (\%) & ($\times$) & & & $p$ & $p$""" + FIN + rf"""
\midrule
{cuerpo}
\bottomrule
\end{{tabular}}
"""
    )


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    pred = json.loads((RESULTS_DIR / "predictions.json").read_text(encoding="utf-8"))
    ev = json.loads((RESULTS_DIR / "evaluation.json").read_text(encoding="utf-8"))

    for nombre, contenido in [
        ("tabla_predicciones.tex", tabla_predicciones(pred)),
        ("tabla_metricas.tex", tabla_metricas(ev)),
    ]:
        (SALIDA / nombre).write_text(contenido, encoding="utf-8")
        print(f"-> {SALIDA / nombre}")


if __name__ == "__main__":
    main()
