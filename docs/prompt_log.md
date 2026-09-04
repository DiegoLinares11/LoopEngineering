# Log del loop asistido por IA

Registro de cómo se construyó el trabajo con asistencia de modelos de lenguaje
(Claude Opus 5 vía Claude Code y Codex de OpenAI). No es una transcripción literal: es el registro de las
decisiones, los descartes y los errores, que es lo que permite juzgar el proceso.

El criterio que ordenó todo el loop fue **no aceptar nada que no se hubiera ejecutado**.
Cada afirmación del paper tiene detrás una corrida cuyo resultado está en `results/`.

---

## 1. Elección del caso: la primera idea no sobrevivió

**Intención inicial:** buscar en Hugging Face un modelo de reconocimiento de imágenes
relacionado con fútbol y explicarlo con SHAP.

**Qué se encontró al buscar:** casi todo lo publicado en Hugging Face bajo "football" son
modelos de **detección** (YOLO para jugadores, balón y árbitro). SHAP para imágenes está
formulado para **clasificación**: necesita una salida escalar por clase. Envolver un
detector para que devuelva un escalar habría obligado a defender en el paper una decisión
arbitraria (¿la confianza de qué caja?) ajena a la pregunta de investigación.

**Decisión:** usar un clasificador preentrenado de ImageNet-1k (`google/vit-base-patch16-224`)
y tratar el fútbol como *caso de prueba*, no como dominio del modelo. ImageNet ya contiene
`soccer ball`, `rugby ball`, `ballplayer` y `scoreboard`, así que la pregunta explicativa se
puede formular sin entrenar nada.

**Lección:** la disponibilidad de modelos condiciona la pregunta que se puede responder.
Conviene verificarla antes de comprometerse con un enunciado.

## 2. Prueba de humo antes de diseñar nada

Antes de escribir una línea del experimento se ejecutó el circuito completo de punta a
punta: descargar ViT → clasificar una imagen → correr el `PartitionExplainer` → guardar una
figura. Confirmó tres cosas que condicionaban el diseño: SHAP corre en CPU en tiempo
razonable (~35 s con 300 evaluaciones), `transformers` y `torch` ya estaban instalados, y
MiKTeX compilaba el preámbulo previsto.

De no haber sido viable en CPU, el diseño habría tenido que reducirse a una sola imagen.

## 3. Diseño experimental: de "una foto de fútbol" a una variable independiente

La primera versión era simplemente "3 fotos de fútbol". Se descartó: tres fotos sin un eje
que las ordene no permiten concluir nada, solo ilustrar.

La versión final fija la clase (`soccer ball`) y varía **una sola cosa**: el área que ocupa
el objeto (69.6 % → 19.9 % → 1.1 %, un factor de 63). Eso convierte tres anécdotas en un
gradiente con una hipótesis falsable.

Las imágenes se eligieron consultando la API de Wikimedia Commons filtrando por licencia,
descargando candidatas y **mirándolas** antes de decidir. Quedaron fijadas por título exacto
en el cuaderno, no por búsqueda, para que la descarga sea determinista.

## 4. Decisiones técnicas y por qué

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| `PartitionExplainer` (caja negra) | `DeepExplainer` / gradientes | Reproduce la condición real de auditar un modelo de terceros y no se ata a la arquitectura |
| Atribuir sobre *logits* | Atribuir sobre *softmax* | Con p = 0.997 la probabilidad está saturada y el mapa se aplana sin que eso signifique ausencia de evidencia |
| Recorte centrado 224 px | Redimensionar deformando | No distorsiona las proporciones de la escena; se verificó mirando los recortes que el balón sobrevive en las tres |
| Desenfoque gaussiano como "borrado" | Gris uniforme / negro | No introduce bordes artificiales que el modelo pueda leer como estructura |

## 5. El giro del trabajo: la concentración no bastaba

La primera métrica planeada era solo la **concentración** de la masa SHAP dentro de la caja
del balón. Al obtener los números apareció el problema: en la imagen B, el 46.9 % de la
evidencia positiva caía fuera del balón. La lectura inmediata era "el contexto pesa casi
tanto como el objeto".

Esa conclusión resultó ser **falsa**, y solo se detectó al añadir las ablaciones: borrar todo
el contexto costaba 0.118 de probabilidad, borrar el balón costaba 0.982. La masa de
atribución estaba sobrestimando el papel causal del contexto.

Añadir las ablaciones no estaba en el plan original y es lo que convirtió el trabajo en un
resultado metodológico en vez de una descripción. Costaron 2 pasadas hacia adelante por
imagen, frente a las 2000 del mapa.

## 6. Errores cometidos y cómo se detectaron

Se registran porque son el tipo de fallo que sobrevive silenciosamente a una revisión rápida.

1. **Cajas mal anotadas.** Las primeras cajas se trazaron a ojo sobre una rejilla y la de la
   imagen B quedó corrida ~18 px a la derecha. Se detectó **dibujándolas sobre la imagen y
   mirándolas**, y se corrigió con un umbral de color. Sin esa verificación visual, la
   métrica de concentración habría sido incorrecta sin dar ninguna señal de error.

2. **El shell se comió un backslash.** Al escribir código Python mediante *heredocs*, la
   secuencia `\times` de una etiqueta LaTeX se convirtió en un carácter TAB literal. La
   figura se generó sin error, solo con la etiqueta corrupta. Se detectó al **mirar el PDF
   compilado**. Corregido escribiendo los archivos directamente en vez de por el shell.

3. **`f-string` en lugar de `rf-string`.** El generador de tablas producía `egin{tabular}`
   en vez de `\begin{tabular}`: Python interpretaba `\b` como *backspace*. Silencioso otra
   vez; se detectó al inspeccionar el `.tex` generado.

4. **Tiempo de ejecución falseado.** La primera corrida completa reportó 37 434 s para una
   imagen: la máquina se suspendió a media noche y el reloj de pared contó la suspensión.
   De haberse copiado al paper, habría dado una cifra de costo absurda. Se cambió a tiempo
   de CPU, con la advertencia de que `process_time` suma todos los hilos.

El patrón es común a los cuatro: **ninguno lanzó un error**. Todos se detectaron mirando la
salida, no ejecutando el código.

## 7. Qué aportó la asistencia de IA y qué no

**Aportó:** velocidad de andamiaje (estructura del repo, scripts, preámbulo LaTeX),
recordar la API de `shap` sin consultar documentación, y proponer las métricas de
verificación estándar (curva de supresión, ablaciones) con sus referencias.

**No aportó, y hubo que imponerlo:** el criterio de no escribir ninguna cifra a mano, la
verificación visual de cada paso intermedio, y la decisión de qué se puede afirmar con tres
imágenes. La tentación por defecto es generar texto que suene concluyente; acotar las
afirmaciones a lo que los datos sostienen fue trabajo de revisión, no de generación.

**Riesgo específico observado:** el modelo produce con la misma fluidez una cifra calculada
y una plausible. La única defensa efectiva fue estructural: generar las tablas desde los
JSON de resultados, de modo que ninguna cifra del documento pueda
provenir de la redacción.

## 8. Reestructuración final: de scripts a un solo cuaderno

El proyecto se desarrolló como seis scripts `.py` bajo `src/`. Al revisar el entregable se
optó por reunir todo en un único `analisis_shap.ipynb`, y conviene dejar por escrito el
razonamiento porque hay una pérdida real.

**A favor del cuaderno:** el enunciado pide "código/notebook SHAP"; un cuaderno se abre y se
lee de arriba abajo sin reconstruir mentalmente el orden de ejecución de seis archivos; y las
figuras quedan junto al texto que las interpreta, que es justo lo que un trabajo sobre
explicabilidad debería hacer bien.

**En contra:** un cuaderno admite estados ocultos —celdas ejecutadas fuera de orden— que son
lo opuesto a la reproducibilidad que el paper reclama. Los scripts obligaban a que cada paso
partiera de un proceso limpio.

**Cómo se mitigó:** el cuaderno se ejecuta completo y en orden con `nbconvert --execute`
antes de cada entrega, de modo que las salidas guardadas provienen siempre de una corrida
única y secuencial, nunca de celdas sueltas. Y se mantuvo lo esencial: el cuaderno sigue
escribiendo los JSON de resultados y generando las tablas `.tex` desde ellos.

**Lo que no se hizo, deliberadamente:** conservar los `.py` *y* el cuaderno. Habría duplicado
la lógica en dos lugares que se desincronizan en cuanto uno cambia — exactamente el defecto
que este trabajo critica cuando pide que las cifras del paper no se transcriban a mano.

## 9. Revisión final del paquete

**Petición:** comparar el repositorio con la lista de entregables y buscar una mejora concreta
antes de cerrar la entrega.

**Hallazgo:** el paquete contenía el paper, el notebook, las figuras, las referencias, el log
y las limitaciones, pero el crédito de `close_up.jpg` quedaba vacío porque Wikimedia Commons
no publica el campo `Artist` de esa fotografía. La página enlaza la publicación original del
usuario `pockethifi` en Flickr; se añadió ese crédito como respaldo reproducible.

**Mejora:** se incorporó `verify_submission.py`, una comprobación sin conexión que valida los
archivos obligatorios, JSON y notebook, hashes y créditos de imágenes, y dependencias LaTeX.
La revisión también sincronizó la autoría de Andy Fuentes en el paper, el PDF, el notebook y
el README. Así, la última revisión deja una prueba ejecutable de integridad en vez de depender
de una inspección manual de carpetas.
