import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

entradas = tf.constant([[0,0], [0,1], [1,0], [1,1]], dtype=tf.float32)
salidas_esperadas = tf.constant([[0], [0], [0], [1]], dtype=tf.float32)

modelo_lineal = tf.layers.Dense(units=1)
salidas_pred = modelo_lineal(entradas)
perdida = tf.losses.mean_squared_error(labels=salidas_esperadas, predictions=salidas_pred)
optimizador = tf.train.GradientDescentOptimizer(0.01)
entrenamineto = optimizador.minimize(perdida)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(150):
        nada, valores_perdida = sess.run((entrenamineto, perdida))
        print(valores_perdida)
    print(sess.run(salidas_pred))
