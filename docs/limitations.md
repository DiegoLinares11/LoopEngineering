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

Todo corre en CPU. Con `EVALS = 2000`, el cálculo de atribuciones tarda del orden de 5
minutos por imagen (~15 min las tres) en un portátil reciente; el resto del cuaderno tarda
menos de dos minutos en total. Con `EVALS = 120` el cuaderno entero corre en menos de dos
minutos y sirve para verificar la instalación antes de la corrida definitiva.

Los tiempos que guarda `results/evaluation.json` son de **CPU** (`process_time`), que suma
el trabajo de todos los hilos de `torch`: son aproximadamente el tiempo de pared
multiplicado por los núcleos ocupados, no lo que hay que esperar frente a la pantalla. Se
midió así porque el tiempo de pared quedaba falseado si la máquina se suspendía a mitad de
la corrida.

## Sensibilidad al presupuesto de evaluaciones

El presupuesto no afecta a todas las cifras por igual, y conviene saber cuáles se mueven.
Comparando una corrida con `EVALS = 120` contra la de `EVALS = 2000`:

| Métrica | 120 evals | 2000 evals | ¿Depende del presupuesto? |
|---|---|---|---|
| Masa positiva en el objeto (A / B / C) | 76.4 / 41.3 / 2.3 % | 83.4 / 53.1 / 3.5 % | **Sí** |
| Enriquecimiento (A / B / C) | ×1.1 / ×2.1 / ×2.1 | ×1.2 / ×2.7 / ×3.2 | **Sí** |
| AUC del borrado aleatorio | 0.704 / 0.294 / 0.042 | idéntico | No |
| Ablaciones (solo balón / sin balón) | 0.994·0.020 / 0.865·0.001 / 0.009·0.058 | idéntico | No |

La razón es que un presupuesto mayor produce una partición más fina, y la masa se
redistribuye dentro de regiones más pequeñas. El orden entre las tres condiciones se
mantiene en ambos casos.

Esto importa para leer el trabajo: **las conclusiones del paper descansan sobre las
ablaciones, que no dependen del presupuesto**. Las métricas de concentración, que sí
dependen, son justamente las que el paper muestra que resultan engañosas.

## Dependencias externas

La sección de datos del cuaderno requiere acceso a la API de Wikimedia Commons. Si un
archivo se renombrara o se retirara de Commons, la descarga fallaría; el manifiesto conserva título,
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
completa está en `data/images/manifest.json`. El código de este repositorio es de los autores;
los pesos del modelo se rigen por la licencia de su publicación en Hugging Face.
