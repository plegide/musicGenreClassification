# Genre Classifier (USO)

## Extraccion de datos (Opcional)

**Este paso NO es necesario para la ejecución, podemos encontrar los archivos.data ya generados para cada aproximación en la carpeta `datasets`**
**Para obtener los mismos resultados obtenidos en la memoria NO hace falta modificarlo**

1. El dataset usado puede ser descargado en
   https://www.kaggle.com/code/andradaolteanu/work-w-audio-data-visualise-classify-recommend/input (La carpeta `genres_original`)

2. Una vez descargado, meteremos en la carpeta `genres` los generos de los que extraer los datos para la creación del archivo .data.

3. Posteriormente ejecutaremos el archivo `split_audio.jl` para extraer los segmentos necesarios por cada pista para el calculo de la FFT y demás características.

4. Una vez tengamos los segmentos, ejecutaremos el archivo `generate_data.jl`, especificando en su configuración el nombre del archivo deseado.

## Ejecución de aproximaciones (RNA, SVM, KNN, DT)

Ejecutaremos el archivo `aproxX.jl` siendo X el número de la aproximación, eligiendo el / los modelo/s a probar cambiando sus hiperparámetros para obtener las distintas arquitecturas. Como se indica en los comentarios de dicho código.

## Deep learning

1. Extraeremos el archivo `segmentos.zip` en la carpeta `segments`
2. Ejecutaremos el archivo `deep_learning.jl`, en este caso para realizar cambios en la arquitectura deberemos editar el archivo `fonts/funciones.jl` donde se encuentra en código fuente, a partir de la línea 695 hasta la línea 708

**Importante: La carpeta segments debe contener únicamente las carpetas extraídas del archivo `segmentos.zip`**

## Generación de gráficas

En este caso, _(ya que no forma parte del código fuente y únicamente es para una mejor visualización y concimiento de los datos)_ usamos python para esta tarea debido a su amplia cantidad de librerías para la creación de gráficas. Para su uso simplemente ejecutaremos el archivo `swarm_plot.py`, y se creará una gráfica por aproximación.

## Cálculo de impacto de métricas

Como en el caso anterior, por motivos similares, usamos python para realizar esta tarea. Su principal funcion es calcular lo capaces que son las métricas de distinguir ciertos géneros.

Para su uso ejecutaremos el archivo `calculate_props.py`, que mostrará por salida estándar en formato LaTeX la media, desviación típica, mínimo y máximo de cada métrica por género.
