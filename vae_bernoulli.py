import math

import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
"""
based on 
https://www.tensorflow.org/tutorials/generative/cvae?hl=pl
"""
def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    predictions = predictions * 0.5 + 0.5
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('test_images\\image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.metric = tf.keras.metrics.Mean(name = "loss")
        self.ELBO_estimate = tf.keras.metrics.Mean(name = "elbo")
        self.encoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid'),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(16, self.latent_dim))
        return self.decoder(eps)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def log_normal_pdf(self, sample, mean, logvar, jacobian = 0.0, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            (-1.0 - jacobian) * logvar - 0.5 * log2pi - 0.5 * (sample-mean) ** 2 * (tf.exp(-logvar) ** 2),
          axis=raxis)

    @tf.function
    def check_result(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_pred = self.decoder(z)
        #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum( tf.keras.losses.BinaryCrossentropy(  reduction= tf.keras.losses.Reduction.NONE)(x, x_pred), axis=[1,2])
        logpz = -0.5 * tf.reduce_sum( tf.square(z) + tf.math.log( 2* math.pi), axis=-1)
        logqz_x =  -0.5 * tf.reduce_sum( tf.square((z-mean)/tf.exp(logvar) ) + tf.math.log( 2* math.pi) + logvar, axis=-1)
        return tf.reduce_mean(logpx_z),  -tf.reduce_mean(logpz - logqz_x)

    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_pred = self.decoder(z)
        #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum( tf.keras.losses.BinaryCrossentropy(  reduction= tf.keras.losses.Reduction.NONE)(x, x_pred), axis=[1,2])
        logpz = -0.5 * tf.reduce_sum( tf.square(z) + tf.math.log( 2* math.pi), axis=-1)
        logqz_x =  -0.5 * tf.reduce_sum( tf.square((z-mean)/tf.exp(logvar) )+ tf.math.log( 2* math.pi) + logvar, axis=-1)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.metric.update_state(loss)
        return {"loss": self.metric.result()}

    @tf.function
    def test_step(self, data):
        loss = -self.compute_loss(data)
        self.ELBO_estimate.update_state(loss)
        return {"elbo":self.ELBO_estimate.result()}

class CreateSample(tf.keras.callbacks.Callback):

    def __init__(self, model, path, image_name_convetion, save_freq = 4):
        super(CreateSample, self).__init__()
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        predictions = model.sample()
        fig = plt.figure(figsize=(30, 30))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('test_images\\image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()


(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))


latent_dim = 2
optimizer = tf.keras.optimizers.Adam(1e-4)
model = CVAE(latent_dim)
model.compile(optimizer)
history  = model.fit(train_dataset,  epochs = 15, callbacks=[ CreateSample(model, None, None)])
model.encoder.summary()
model.decoder.summary()

logp_mean = 0
kl_mean =0
i = 0
'''
for batch in test_dataset:
    logp, kl = model.check_result(batch)
    logp_mean += logp
    kl_mean += kl
    i = i +1

print(logp_mean/i)
print(kl_mean/i)
print(logp_mean/i - kl_mean/i)

'''



digit_size = 28
n = 20
norm = tfp.distributions.Normal(0, 1)
grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
image_width = digit_size * n
image_height = image_width
image = np.zeros((image_height, image_width))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z = np.array([[xi, yi]])
        x_decoded = model.sample(z)
        x_decoded = 0.5 *x_decoded + 0.5
        digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
        image[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit.numpy()

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='Greys_r')
plt.axis('Off')
plt.show()

