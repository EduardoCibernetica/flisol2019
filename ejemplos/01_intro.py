import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

#Tensores: "La unidad central principal de TensorFlow es el tensor".
variable1 = tf.constant(2, )
variable2 = tf.constant(3)
variable3 = tf.constant([1,2,3])
variable4 = tf.constant([[1,2,3],[1,2,3]])
variable5 = tf.constant([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
# Grafico "Un gráfico computacional es una serie de operaciones".
# 1 - Los nodos de la gráfica: las operaciones que producen Tensores.
# 2 - Los bordes en la gráfica: Valores que fluiran a través del  gráfico.

variable_a = tf.constant(1)
variable_b = tf.constant(4)
suma = tf.add(variable_a, variable_b)
resta = tf.subtract(suma, variable2)

#TensorBoard: Proporciona una utilidad para visualizar un gráfico de cómputo.
# 1 - Ejecute tensorboard --logdir .
# 2 - http://localhost:6006
guardar = tf.summary.FileWriter('.')
guardar.add_graph(tf.get_default_graph())
guardar.flush()

# Sesión: "Ejecuta las operaciones de TensorFlow"
with tf.Session() as sess:
    salida = sess.run(suma)
    print(salida)

#Feeding Para introducir datos externos en un gráfico.
valor_x = tf.placeholder(tf.float32)
valor_y = tf.placeholder(tf.float32)
valor_z = tf.add(valor_x, valor_y)

with tf.Session() as sess:
    salida = sess.run(valor_z, {valor_x: 2, valor_y: 6})
    print(salida)


#Capas: Las capas agrupan las variables salidas las operaciones que actúan sobre ellas.
#1 - Los pesos salidas los sesgos son gestionados por el objeto de capa.
variable_salida = tf.placeholder(tf.float32, shape=[None, 2])
modelo_lineal = tf.layers.dense(variable_salida, units=1)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    salida = sess.run(modelo_lineal, feed_dict={variable_salida: [[1, 2], [5, 6]]})
    print(salida)

