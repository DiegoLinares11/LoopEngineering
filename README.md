# ¿El modelo mira el balón o mira el césped?

Auditoría con SHAP de un clasificador ViT sobre escenas de fútbol. Primera tarea del curso
de Inteligencia Artificial Responsable: construir, mediante un *loop* asistido por IA, un
paper en LaTeX **reproducible** que explique las predicciones de un clasificador de
imágenes.

**Autores:** Andy Fuentes y Diego Linares.

**Análisis:** [`analisis_shap.ipynb`](analisis_shap.ipynb) · **Paper:**
[`paper/paper.pdf`](paper/paper.pdf) · **Log del loop:**
[`docs/prompt_log.md`](docs/prompt_log.md)

## La pregunta

Cuando el modelo responde `soccer ball` ante una escena de fútbol, ¿la evidencia está en el
balón o en el escenario correlacionado (césped, jugadores, gradería)?

El diseño fija la clase y varía **una sola cosa**: el área que ocupa el objeto, a lo largo
de dos órdenes de magnitud.

| | Área del balón | p(`soccer ball`) |
|---|---|---|
| **A.** Close-up | 69.6 % | 0.997 |
| **B.** Balón en el campo | 19.9 % | 0.983 |
| **C.** Escena de partido | 1.1 % | 0.104 |

Modelo: `google/vit-base-patch16-224` (ViT-B/16, ImageNet-1k), sin ajuste fino y tratado
como caja negra.

## El resultado

Las atribuciones SHAP no se interpretan mirándolas: se someten a pruebas que pueden
refutarlas.

1. **SHAP es fiel.** Borrar píxeles en orden de atribución degrada la predicción mucho más
   rápido que borrarlos al azar, en los tres casos (AUC 0.291 vs 0.704 en A).
2. **La masa de atribución no mide necesidad causal.** En B, el 46.9 % de la evidencia cae
   fuera del balón, pero borrar todo el contexto cuesta 0.118 de probabilidad mientras que
   borrar el balón cuesta 0.982.
3. **En C la predicción correcta no depende del balón.** Conservar solo el balón deja
   p = 0.009 (no basta); borrarlo deja p = 0.058 frente al 0.104 original (no es
   necesario). El modelo reconoce una escena de fútbol, no un balón de fútbol.

La conclusión: ningún mapa de atribución debería reportarse como explicación sin una prueba
de necesidad y una de suficiencia sobre la región que señala.

## Reproducir

Todo el análisis vive en **un solo cuaderno autocontenido**, que no depende de ningún otro
archivo del repositorio: descarga las imágenes, clasifica, calcula SHAP, corre las tres
pruebas de verificación y escribe los resultados, las figuras y las tablas del paper.

```bash
jupyter notebook analisis_shap.ipynb     # Ejecutar > Ejecutar todas las celdas
```

El cuaderno instala por su cuenta lo que falte, así que también corre en Colab sin clonar
el repositorio. Requiere Python 3.11+.

El parámetro `EVALS` controla el presupuesto de evaluaciones de SHAP: con **2000** se
obtienen los mapas del paper (~5 min por imagen en CPU); con **120** el cuaderno entero
corre en menos de dos minutos y sirve para comprobar que todo funciona.

Después, para reconstruir el PDF:

```bash
cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Antes de entregar, el paquete completo se valida sin conexión con:

```bash
python verify_submission.py
```

El verificador comprueba los entregables, la estructura del notebook y los JSON, los hashes
de las imágenes, sus créditos y todas las figuras, tablas y referencias usadas por LaTeX.

Las semillas están fijas y cada JSON de resultados registra las versiones exactas de
Python, `torch`, `transformers`, `shap` y `numpy` con las que se produjo. Las tablas del
paper se generan desde esos JSON: ninguna cifra del documento está escrita a mano.

## Estructura

```
analisis_shap.ipynb   el análisis completo — única fuente de verdad
verify_submission.py  verificación automática del paquete de entrega
data/                 imágenes de prueba, manifiesto de licencias y cajas anotadas
results/              salidas en JSON y valores SHAP crudos (.npz)
figures/              figuras generadas que consume el paper
paper/                paper.tex, bibliografía, tablas generadas y PDF compilado
docs/                 log del loop y limitaciones de ingeniería
```

## Licencias

Las imágenes provienen de Wikimedia Commons bajo licencias libres (dos CC0 y una CC BY
2.0). `data/images/manifest.json` guarda título, autor, licencia, URL y `sha256` de cada
archivo; la atribución se reproduce en el paper.
