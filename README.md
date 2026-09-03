# LoopEngineering — Paper reproducible sobre SHAP

Primera tarea del curso: construir, mediante un *loop* asistido por IA, un paper en LaTeX
**reproducible** que explique las predicciones de un clasificador de imágenes usando SHAP.

## Caso de estudio

Modelo preentrenado `google/vit-base-patch16-224` (ViT-B/16, ImageNet-1k, 1000 clases) aplicado a
fotografías de fútbol. ImageNet incluye clases directamente relevantes: `soccer_ball`, `ballplayer`,
`rugby_ball`, `basketball`, `scoreboard`.

**Pregunta explicativa:** cuando el modelo clasifica una escena de fútbol, ¿la evidencia se concentra
en el objeto que nombra la clase (el balón) o en el contexto correlacionado (césped, uniformes,
gradería)? Es decir, ¿la predicción se sostiene en la señal causal o en un atajo espurio del dataset?

## Estructura

```
src/        scripts reproducibles (descarga, clasificación, SHAP)
data/       imágenes de prueba + manifiesto de licencias y procedencia
results/    salidas en JSON (predicciones, métricas de atribución)
figures/    figuras generadas que consume el paper
paper/      paper.tex, bibliografía y PDF compilado
docs/       log de prompts y limitaciones
```

## Reproducir

Requiere Python 3.11+ y una distribución de LaTeX con `pdflatex`.

```bash
pip install -r requirements.txt
python src/fetch_images.py      # descarga las imágenes y escribe el manifiesto
python src/classify.py          # predicciones top-5 -> results/predictions.json
python src/explain_shap.py      # atribuciones SHAP -> figures/ y results/
cd paper && pdflatex paper.tex
```

Todo script fija semillas y registra las versiones de las librerías en su salida JSON, de modo que
las cifras del paper puedan rastrearse hasta la corrida que las produjo.

## Licencias

Las imágenes provienen de Wikimedia Commons bajo licencias libres. `data/images/manifest.json`
guarda título, autor, licencia y URL de cada archivo; la atribución se reproduce en el paper.
