import numpy
import numpy as np
import tensorflow as tf

from src.display import show_two_images_3D

m = tf.keras.models.load_model("/Users/ducanhnguyen/Documents/testingforAI-vnuuet/vae/mnist_50k_5kepochs_latent8")
m.summary()

decoder = m.layers[1]

latent_dim_len = decoder.inputs[0].shape[1]

while True:
    input = numpy.random.normal(loc=0.0, scale=1.0, size = latent_dim_len)
    print(input)
    input = input.reshape(-1, latent_dim_len)

    out = decoder.predict(input)

    show_two_images_3D(out[0], out[0], display=True)


