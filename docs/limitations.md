# Limitaciones

Las limitaciones que afectan a las conclusiones del paper están argumentadas en su
sección 5. Aquí se registran las de nivel de ingeniería: lo que un tercero necesita saber
para reproducir o extender el trabajo sin sorpresas.

## Alcance de lo que se afirma

Tres imágenes ilustran un mecanismo; no estiman su frecuencia. Ninguna afirmación del
paper se generaliza a "los modelos de visión" ni a "las escenas de fútbol": todas se
enuncian sobre estos tres casos y sobre este modelo.

En particular, **no se afirma que exista sesgo geográfico**. La imagen C fue tomada en
Uganda y es la que peor clasifica, pero el diseño manipula el tamaño del objeto, que
explica suficientemente la diferencia. Separar ambos factores exigiría un conjunto
estratificado por región y por tamaño, que es la extensión natural del trabajo.

## Reproducibilidad: qué se garantiza y qué no

**Se garantiza:** las mismas imágenes (fijadas por título en Commons y verificadas por
`sha256`), el mismo modelo (pesos públicos y versionados), las mismas semillas
(`20260902`) y las mismas versiones de librería, registradas dentro de cada JSON de
resultados.

**No se garantiza la igualdad bit a bit entre máquinas.** Las operaciones de punto
flotante de `torch` dependen del número de hilos, del conjunto de instrucciones de la CPU
y de la versión de las librerías de álgebra lineal. Las diferencias esperables afectan a
los últimos decimales y no al orden de magnitud ni al sentido de las comparaciones, que
es lo que sostienen las conclusiones.

`torch.use_deterministic_algorithms(True, warn_only=True)` está activo, pero en modo
`warn_only`: si alguna operación no tiene implementación determinista, avisa en vez de
abortar. Se eligió así para que el script no falle en entornos distintos al de desarrollo.

## Costo de cómputo

Todo corre en CPU. Con 2000 evaluaciones por imagen, `explain_shap.py` tarda del orden de
5 minutos por imagen (~15 min las tres) en un portátil reciente. `evaluate.py` y
`classify.py` tardan menos de un minuto cada uno.

Para iterar sobre el diseño de la figura existe `python src/explain_shap.py --solo-figura`,
que la redibuja desde `results/shap_values.npz` sin recalcular las atribuciones. El
presupuesto de evaluaciones se controla con `--evals`; con 120 el circuito completo corre
en menos de un minuto y sirve para verificar que todo funciona antes de la corrida final.

Los tiempos que guarda `results/shap_summary.json` son de **CPU** (`process_time`), que
suma el trabajo de todos los hilos de `torch`: son aproximadamente el tiempo de pared
multiplicado por los núcleos ocupados, no lo que hay que esperar frente a la pantalla. Se
midió así porque el tiempo de pared quedaba falseado si la máquina se suspendía a mitad de
la corrida.

## Dependencias externas

`fetch_images.py` requiere acceso a la API de Wikimedia Commons. Si un archivo se
renombrara o se retirara de Commons, la descarga fallaría; el manifiesto conserva título,
URL, autor, licencia y `sha256` de cada imagen, de modo que el conjunto es identificable
aunque haya que recuperarlo por otra vía.

Los pesos de `google/vit-base-patch16-224` se descargan de Hugging Face en la primera
ejecución (~350 MB) y quedan en la caché local.

## Anotaciones

Las cajas de `data/annotations.json` son **cajas envolventes**, no segmentaciones, y
fueron anotadas por el autor. Incluyen algo de fondo en las esquinas, lo que sobreestima
la evidencia atribuida al objeto y hace conservadora la medida de concentración respecto
de la conclusión que se defiende. Una segmentación precisa cambiaría los valores absolutos
de la columna *Masa*; el orden entre las tres condiciones no depende de ese detalle.

## Licencias

Las imágenes son CC0 (B: `Adidas soccer ball on a grass pitch`; C: `Players and referees
before a football match`) y CC BY 2.0 (A: `Soccer Ball (42232038211)`). La atribución
completa está en `data/images/manifest.json`. El código de este repositorio es del autor;
los pesos del modelo se rigen por la licencia de su publicación en Hugging Face.
