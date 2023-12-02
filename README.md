# Enunciado
El objetivo de esta práctica es construir un proyecto de ML, al que denominaremos clasificador1
para predecir si una imagen tomada de masa mamaria y digitalizada se corresponde con un
diagnóstico benigno (malignant = 0) o maligno (malignant = 1)

La imagen no está disponible. En su lugar hay 8 tablas donde se recogen diferentes características
visuales extraídas con técnicas de visión artificial: traintab01.csv, ... , traintab08.csv.

El proyecto debe ser tal que se pueda poner en producción a partir de su entrega. Esto significa
que, una vez entregado, el cliente puede probar nuevos ejemplos con un interfaz mínimo: simplemente proporcionando los ejemplos cumpliendo con el formato de las tablas e invocando un
script de Python para generar un fichero de etiquetas estimadas.

Una vez cerrada la entrega, se realizará una competición entre todos los proyectos con un conjunto
de tablas reservado.

# Funcionamiento

Dentro del fichero `train_model.py` tenemos los datos para entrenar el modelo. Al entrenar el modelo usamos los datos de entrenamiento que tenemos dentro del código (hay que cambiar los paths para datos nuevos), cuando este modelo es entrenado lo guardamos en un archivo .joblib y es evaluados con el conjunto de test testabs. Todos estos paths pueden cambiarse en el código.

Dentro de este fichero podemos seleccionar que modelo queremos entrenar y ajustar los hiperparámetros a entrenar.

El comando para ejecutar el código es:

~~~
python3 train_model.py
~~~

Una vez entrenado el modelo este será evaluado nada más entrenarse generando el fichero `Competicion.txt`, sin embargo, podremos usar el modelo nuevamente sin tener que entrenarlo de nuevo puesto que se guarda en el fichero .joblib.

Para usarlo tenemos que utilizar el siguiente comando:


~~~
python3 load_model.py -f path_tab1,...,path_tab8 -p path_modelo.joblib
~~~

Y esto utilizará el modelo del joblib sobre esos tabs.

Por último, cuando entrenamos un modelo y lo ejecutamos sobre nuestro conjunto de pruebas registramos su accuracy, las componentes de la pipeline con sus hiperparámetros en el fichero `metricas_obtenidas.csv` y el momento en el que se hizo evaluo la prueba.

