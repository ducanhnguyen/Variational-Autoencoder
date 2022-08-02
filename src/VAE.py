import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.models import Model

from src.GaussianSampling import GaussianSampling
from src.display import show_two_images_3D
from matplotlib import pyplot as plt


class VAE(Model):
    def __init__(self, input_shape, latent_dim=16, **kwargs):
        super(VAE, self).__init__(**kwargs)
        height, width, channels = input_shape

        '''
        ENCODER
        '''
        dense_depth = 0
        conv_depth = 0
        last_dense_shape = None
        last_conv_shape = None

        encoder_inputs = keras.Input(shape=(height, width, channels))
        x = encoder_inputs

        while True:
            if x.shape[1] < 10 or x.shape[2] < 10:  # height or width
                break

            conv_depth += 1
            # *2 the number of channels while /2 the size of feature maps
            x = layers.Conv2D(filters=x.shape[3] * 4,  # * 2 the number of channels
                              kernel_size=3,
                              activation="relu",
                              strides=2,  # /2 width and height of feature maps
                              padding="same"
                              )(x)
            last_conv_shape = x.shape

        x = layers.Flatten()(x)

        while True:
            if x.shape[1] < 256 or x.shape[1] < latent_dim * 2:
                break
            x = layers.Dense(x.shape[1] / 2, activation="relu")(x)
            dense_depth += 1
            last_dense_shape = x.shape

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        z = GaussianSampling()(mean=z_mean,
                               log_variance=z_log_var)

        encoder = keras.Model(inputs=encoder_inputs,
                              outputs=[z_mean, z_log_var, z],
                              name="encoder")
        encoder.summary()

        '''
        DECODER
        '''
        latent_inputs = keras.Input(shape=(latent_dim,))

        x = layers.Dense(last_dense_shape[1], activation="relu")(latent_inputs)

        while dense_depth > 0:
            x = layers.Dense(x.shape[1] * 2, activation="relu")(x)
            dense_depth -= 1

        x = layers.Reshape((last_conv_shape[1], last_conv_shape[2], last_conv_shape[3]))(x)

        while conv_depth > 0:
            if conv_depth == 1:
                x = layers.Conv2DTranspose(filters=x.shape[3] / 4,
                                           kernel_size=3,
                                           activation="sigmoid",
                                           strides=2,
                                           padding="same")(x)
            else:
                x = layers.Conv2DTranspose(filters=x.shape[3] / 4,
                                           kernel_size=3,
                                           activation="relu",
                                           strides=2,
                                           padding="same")(x)
            conv_depth -= 1

        decoder = keras.Model(inputs=latent_inputs,
                              outputs=x,
                              name="decoder")
        decoder.summary()

        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        _mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            kl_loss = 0.5 * (tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__ == '__main__':
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    # (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    MODEL_PATH = '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/vae/cifar10_50k'

    if len(x_train.shape) == 3:  # Ex: (batch, height, width)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

    if len(x_test.shape) == 3:  # Ex: (batch, height, width)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    x_train = x_train.astype("float32") / 255

    x_train = x_train[:None]
    vae = VAE(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
              latent_dim=16)
    input = tf.zeros(shape=(1, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    output = vae(input)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    history = vae.fit(x_train, epochs=2000, batch_size=1024)
    vae.save(MODEL_PATH)

    plt.plot(history.history['kl_loss'])
    plt.ylabel('kl_loss')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['reconstruction_loss'])
    plt.ylabel('reconstruction_loss')
    plt.xlabel('epoch')
    plt.show()

    res = vae.predict(x_train)
    for idx in range(0, 10):
        show_two_images_3D(x_train[idx], res[idx], display=True)
