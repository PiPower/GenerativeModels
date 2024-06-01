import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import argparse
import matplotlib.pyplot as plt
import os
import random


def process_path(file_path, x_res=80, y_res=80):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [x_res, y_res])
    img = img / 127.5 - 1
    return img


def ResblockDown(x, hidden_dim, iteration=0, depth=1):
    input_tensor = tf.keras.layers.Input(x.shape[1:])
    y = tf.keras.layers.Conv2D(hidden_dim, (1, 1))(input_tensor)

    for it in range(depth):
        x = tf.keras.layers.BatchNormalization()(y)
        x = tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation="relu")(x)
        x = tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation="relu")(x)
        y = x + y

    x = tf.keras.layers.AveragePooling2D()(x)
    return tf.keras.models.Model(inputs=input_tensor, outputs=[x], name="resblock_down_{0}".format(iteration))


def ResblockUp(x, hidden_dim, iteration=0, depth=1):
    input_tensor = tf.keras.layers.Input(x.shape[1:])
    y = tf.keras.layers.UpSampling2D()(input_tensor)
    y = tf.keras.layers.Conv2D(hidden_dim, (1, 1))(y)

    for it in range(depth):
        x = tf.keras.layers.BatchNormalization()(y)
        x = tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation="relu")(x)
        x = tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation="relu")(x)
        y = x + y

    return tf.keras.models.Model(inputs=input_tensor, outputs=[y], name="resblock_up_{0}".format(iteration))


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, depth=1, hidden_dim_enc=[64, 256, 512, 1024],
                 hidden_dim_dec=[1024, 512, 256, 64], start_res=4, entry_shape=(64, 64, 3)):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.depth = depth
        self.entry_input = entry_shape
        self.hidden_dim_enc = hidden_dim_enc
        self.hidden_dim_dec = hidden_dim_dec
        self.start_res = start_res
        self.opt_schedule = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.metric = tf.keras.metrics.Mean(name="loss")
        self.ELBO_estimate = tf.keras.metrics.Mean(name="elbo")
        input = tf.keras.layers.Input(self.entry_input)
        x = input

        for i in range(len(hidden_dim_enc)):
            x = ResblockDown(x, hidden_dim_enc[i], i, depth)(x)

        flat = tf.keras.layers.Flatten()(x)
        mean = tf.keras.layers.Dense(latent_dim, name="mean")(flat)
        logvar = tf.keras.layers.Dense(latent_dim, name="logvar")(flat)
        L = tf.keras.layers.Dense(latent_dim ** 2, name="l-matrix")(flat)
        L = tf.keras.layers.Reshape((latent_dim, latent_dim))(L)

        self.encoder = tf.keras.models.Model(inputs=input, outputs=[mean, logvar, L])

        input_dec = tf.keras.layers.Input((latent_dim,))
        x = tf.keras.layers.Dense(units=start_res * start_res * hidden_dim_dec[0], activation=tf.nn.relu)(input_dec)
        x = tf.keras.layers.Reshape((start_res, start_res, hidden_dim_dec[0]))(x)
        for i in range(len(hidden_dim_dec)):
            x = ResblockUp(x, hidden_dim_dec[i], i, depth)(x)

        out_dec = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x)
        self.decoder = tf.keras.models.Model(inputs=input_dec, outputs=out_dec)

    def get_config(self):
        return {"latent_dim": self.latent_dim,
                "depth": self.depth,
                "hidden_dim_enc": self.hidden_dim_enc,
                "hidden_dim_dec": self.hidden_dim_dec,
                "start_res": self.start_res,
                "entry_input": self.entry_input
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        instance = cls(**config)
        return instance

    def conditional_sample(self, img):
        mean, logvar, L = self.encoder(img)
        z, eps = self.reparameterize(mean, logvar, L)
        x_hat = self.decoder(z)
        return x_hat

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(16, self.latent_dim))
        return self(eps)

    def call(self, inputs, training=None, mask=None):
        return self.decoder(inputs)

    @tf.function
    def reparameterize(self, mean, logvar, L):
        mask = tf.linalg.band_part(tf.ones((self.latent_dim, self.latent_dim)), -1, 0)
        mask = tf.linalg.set_diag(mask, tf.zeros((self.latent_dim)))
        eps = tf.random.normal(shape=(self.latent_dim, 1))
        L = tf.linalg.set_diag(mask * L, tf.exp(logvar))
        x = tf.matmul(L, eps)
        x = tf.squeeze(x, axis=-1)
        return x + mean, tf.squeeze(eps, axis=-1)

    @tf.function
    def compute_loss(self, x):
        log2pi = tf.math.log(2. * np.pi)
        mean, logvar, L = self.encoder(x)
        z, eps = self.reparameterize(mean, logvar, L)
        x_hat = self.decoder(z)
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        log_q_z = -0.5 * tf.reduce_sum((tf.math.square(eps) + log2pi + logvar), axis=-1)
        log_p_z = -0.5 * tf.reduce_sum((tf.square(z) + log2pi), axis=-1)
        log_p_x = -0.5 * tf.reduce_sum(tf.square(x_hat - x), axis=[1, 2, 3])
        return -tf.reduce_mean(log_p_x + self.opt_schedule * (log_p_z - log_q_z))

    @tf.function
    def check_results(self, x):
        log2pi = tf.math.log(2. * np.pi)
        mean, logvar, L = self.encode(x)
        z, eps = self.reparameterize(mean, logvar, L)
        x_hat = self.decoder(z)
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        log_q_z = -0.5 * tf.reduce_sum((tf.math.square(eps) + log2pi + logvar), axis=-1)
        log_p_z = -0.5 * tf.reduce_sum((tf.square(z) + log2pi), axis=-1)
        log_p_x = -0.5 * tf.reduce_sum(tf.square(x_hat - x), axis=[1, 2, 3])

        return -tf.reduce_mean(log_p_x), -tf.reduce_mean(log_p_z - log_q_z)

    @tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric.update_state(loss)
        return {"loss": self.metric.result()}

    @tf.function
    def test_step(self, data):
        loss = -self.compute_loss(data)
        self.ELBO_estimate.update_state(loss)
        return {"elbo": self.ELBO_estimate.result()}


class CreateSample(tf.keras.callbacks.Callback):

    def __init__(self, model, manager, save_freq=4, larger_sample_freq=None):
        super(CreateSample, self).__init__()
        self.model = model
        self.epsilon = tf.random.normal(shape=(16, self.model.latent_dim))
        self.save_freq = save_freq
        self.checkpoint = manager
        self.larger_sample_freq = save_freq * larger_sample_freq

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.save_freq != 0:
            return
        self.checkpoint.save()

        print("\n")

        predictions = self.model.sample()
        predictions = predictions / tf.math.reduce_max(tf.abs(predictions))
        predictions = 0.5 * predictions + 0.5
        fig = plt.figure(figsize=(30, 30))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i])
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('test_images\\image_at_epoch_{0}_it_0.png'.format(epoch + 1))
        plt.close()

        if self.larger_sample_freq is not None and (epoch + 1) % self.larger_sample_freq == 0:
            val_size = 4
            for j in range(val_size):
                predictions = self.model.sample()
                predictions = predictions / tf.math.reduce_max(tf.abs(predictions))
                predictions = 0.5 * predictions + 0.5
                fig = plt.figure(figsize=(30, 30))

                for i in range(16):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(predictions[i])
                    plt.axis('off')

                # tight_layout minimizes the overlap between 2 sub-plots
                plt.savefig('test_images\\test_sample_{0}_it_{1}.png'.format(j, j + 1))
                plt.close()


class AlphaScheduler(tf.keras.callbacks.Callback):

    def __init__(self, alpha_steps, model, incr_freq=1, **kwargs):
        super(AlphaScheduler, self).__init__(**kwargs)
        self.step_size = 1 / alpha_steps
        self.alpha = tf.Variable(0.0001)
        self.const = tf.Variable(1.0)
        self.model = model
        self.incr_freq = incr_freq

    def on_epoch_begin(self, epoch, logs=None):
        to_assign = tf.minimum(self.const, self.alpha)
        self.model.opt_schedule.assign(to_assign)
        if (epoch + 1) % self.incr_freq == 0:
            self.alpha.assign_add(self.step_size)


def train_on_face(checkpoint_path="./checkpoint/test", restore=False, load_path="./", init_epoch=None):
    dataPath = "C:\\Datasets\\Celeba_HQ"
    batch_size = 64

    filenames = [os.path.join(dataPath, i) for i in os.listdir(dataPath)]
    random.shuffle(filenames)

    Gen_Temp = tf.data.Dataset.from_tensor_slices(filenames[0: int(len(filenames) * 0.85)])
    train_dataset = Gen_Temp.shuffle(buffer_size=1024 * 4, reshuffle_each_iteration=True) \
        .map(lambda x: process_path(x, 32, 32), num_parallel_calls=4) \
        .batch(batch_size, True).prefetch(tf.data.AUTOTUNE).cache()

    Gen_Temp = tf.data.Dataset.from_tensor_slices(filenames[int(len(filenames) * 0.85):])
    test_dataset = Gen_Temp.shuffle(buffer_size=1024 * 4, reshuffle_each_iteration=True) \
        .map(lambda x: process_path(x, 32, 32), num_parallel_calls=4) \
        .batch(batch_size, True).prefetch(tf.data.AUTOTUNE).cache()

    latent_dim = 80
    epochs = 1000
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = CVAE(latent_dim, 1, entry_shape=(64, 64, 3), start_res=4)
    model.compile(optimizer)
    initial_epoch = 0
    if init_epoch:
        initial_epoch = init_epoch

    checkpoint = tf.train.Checkpoint(model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=1)

    if restore:
        checkpoint.restore(load_path)

    model.encoder.summary()
    model.decoder.summary()
    xd = model.sample()
    model.summary()
    callbacks = [CreateSample(model, manager, save_freq=600, larger_sample_freq=1), AlphaScheduler(150, model, 3)]
    model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=test_dataset,
              initial_epoch=initial_epoch)

    tf.keras.models.save_model(model, "saved_vae_multivariate_model")
    val_size = 4
    dataset_iter = iter(train_dataset)
    for j in range(val_size):
        sample = dataset_iter.get_next()[0]
        fig = plt.figure(figsize=(30, 30))
        plt.subplot(4, 4, 1)
        plt.imshow(sample * 0.5 + 0.5)
        plt.axis('off')

        sample = tf.expand_dims(sample, axis=0)
        sample = tf.tile(sample, [15, 1, 1, 1])
        predictions = model.conditional_sample(sample)
        predictions = predictions / tf.math.reduce_max(tf.abs(predictions))

        predictions = 0.5 * predictions + 0.5
        for i in range(1, 16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i - 1])
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('test_images\\test_sample_conditional_{0}.png'.format(j))
        plt.close()

        plt.close()


if __name__ == "__main__":
    checkpoint_path = "./checkpoint"
    restore = False
    restore_path = None
    initial_epoch = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restore", action='store')
    parser.add_argument("-ie", "--initial_epoch", action='store', type=int)
    args = parser.parse_args()

    if args.restore:
        restore = True
        restore_path = args.restore
    if args.initial_epoch:
        initial_epoch = args.initial_epoch

    train_on_face(checkpoint_path, restore, restore_path, initial_epoch)
