import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import math
import functools
import os
import argparse
import  new_test_model
import sde


def process_path(file_path, x_res = 80, y_res = 80):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [y_res, x_res])
    img = img / 255.0
    return img

def plot_images(images, rows, cols, index=0, mode = "rgb", save = True):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    if mode == "gray":
        images = tf.tile(images, [1,1,1,3])
    for y in range(rows):
        for x in range(cols):
            axs[y, x].imshow(images[y * rows + x, :, :, :])
    save_string = './test_images/sde_model_{0}.png'.format(index)
    if save:
        plt.savefig(save_string)
    else:
        plt.show()
    plt.close()

def load_mnist_dataset(elements = None):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.concatenate([train_images, test_images], 0 )
    train_images = train_images.astype('float32') /255.0
    train_images = np.reshape(train_images, newshape=(-1, 28, 28,1))
    if elements is not None and elements < train_images.shape[0]:
        return train_images[0:elements, :, : ]
    return train_images

class GaussianFourierProjection(tf.keras.layers.Layer):
    def __init__(self, embbde_dim, std_dev):
        super(GaussianFourierProjection, self).__init__()
        rand = tf.random.normal(shape=(1, embbde_dim//2)) * std_dev
        self.w = tf.Variable(rand, trainable=False)

    @tf.function
    def call(self, inputs, *args, **kwargs):
        proj = self.w * inputs * 2 * 3.1415926
        out = tf.concat( [tf.sin(proj), tf.cos(proj)], axis=-1)
        return out
@tf.function
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = tf.math.log(float(max_positions))/ (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.reshape(emb, (1, -1))
    timesteps = tf.cast(timesteps, dtype=tf.float32 )
    emb = timesteps * emb
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)

    return emb

class SdeModel(tf.keras.Model):
    def __init__(self, shape, channels, sde, drop_out_rate=0.1, hiddens = [32,64, 86], embedd_dim = 128, depth = 2, scale_by_sigma = False):
        super(SdeModel, self).__init__()

        self.model = new_test_model.get_network((shape[1], shape[0]), hiddens, depth, embedd_dim, channels,False, True, embedd_dim)
        self.hiddens = hiddens
        self.depth = depth
        self.scale_by_sigma = scale_by_sigma
        self.channels = channels
        self.embedd_dim = embedd_dim
        self.sde = sde
        self.timestep_embb = functools.partial(get_timestep_embedding, embedding_dim = embedd_dim)
        self.drop_out_rate = drop_out_rate
        self.channels = channels
        self.shape = (shape[1], shape[0], shape[2] )
        self.gaussian_fourier_proj = GaussianFourierProjection(embedd_dim, 16)
        self.mean_loss = tf.keras.metrics.Mean(name="mean_loss")

    def get_config(self):
        return {
                "layer_config":{
                    "shape": self.shape,
                    "channels": self.channels,
                    "sde": self.sde,
                    "drop_out_rate": self.drop_out_rate,
                    "depth":self.depth,
                    "hiddens":self.hiddens,
                    "embedd_dim":self.embedd_dim
                               },
                }
    @classmethod
    def from_config(cls, config, custom_objects=None):
        instance =  cls(**config["layer_config"])
        return instance

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x, t = inputs
        t_in = self.gaussian_fourier_proj(t)
        #t_in = self.timestep_embb(t)
        img =  self.model([t_in, x], training, mask)
        if self.scale_by_sigma:
            _, norm_factor = self.sde.distribution_params(img, t)
            img = img / norm_factor
        return img

    def euler_maruyama_step(self, x, t, dt, batch_size):
        time_step = tf.ones((batch_size, 1)) * t
        f, g = self.sde.coefficients(x, t)
        drift = f - (g ** 2) * self([x, time_step])
        diffusion = g

        x_mean = x + drift * dt
        x = x_mean + diffusion * tf.sqrt(-dt) * tf.random.normal(tf.shape(x))
        return x, x_mean

    def langevin_step(self,  x, t, snr, batch_size):
        time_step = tf.ones((batch_size, 1)) * t
        if isinstance(self.sde, sde.VpSDE):
            timestep_index = (t * (self.sde.N - 1) / self.sde.T)
            timestep_index = tf.cast(timestep_index, dtype=tf.int32)
            alpha = tf.gather(self.sde.alphas, timestep_index)
        else:
            alpha = tf.ones((1), dtype=tf.float32)
        # Corrector step (Langevin MCMC)
        grad = self([x, time_step])
        noise = tf.random.normal(tf.shape(x))

        grad_norm = tf.reduce_mean(tf.norm(tf.reshape(grad, (batch_size, -1)), axis=-1))
        noise_norm = tf.reduce_mean(tf.norm(tf.reshape(noise, (batch_size, -1)), axis=-1))
        langevin_step_size = 2 * alpha * (snr * noise_norm / grad_norm) ** 2
        x = x + langevin_step_size * grad + tf.sqrt(2 * langevin_step_size) * noise
        return x, None

    def Euler_Maruyama_sampler(self, batch_size=64, eps=1e-3, num_steps = 1000):
        x_prior =  self.sde.prior_sampling((batch_size, ) + self.shape )
        time_steps = tf.linspace(1., eps, num_steps)
        dt = -1.0/ num_steps
        x = x_prior
        for i in range(num_steps):
            if i % 100 == 0:
                print("iter nr: {0}". format(i))
            t = tf.gather(time_steps, i)
            x, x_mean = self.euler_maruyama_step(x, t, dt, batch_size)
        return x_mean

    def pc_sampler(self,
                   batch_size=16,
                   num_steps=500,
                   snr= 0.16,
                   eps=1e-4):
        dt = -1.0 / num_steps
        time_steps = tf.linspace(1., eps, num_steps)
        x =  self.sde.prior_sampling((batch_size, ) + self.shape )
        for i in range(num_steps):
            if i % 100 == 0:
                print("iter nr: {0}".format(i))

            t = tf.gather(time_steps, i)
            x, x_mean = self.langevin_step(x, t, snr, batch_size)
            x, x_mean = self.euler_maruyama_step(x, t, dt, batch_size)

            # The last step does not include any noise
        return x_mean

    @tf.function
    def train_step(self, x_0):
        batch_size = tf.shape(x_0) * tf.convert_to_tensor([1, 0, 0, 0])
        batch_size = tf.reduce_sum(batch_size)

        t = tf.random.uniform((batch_size, 1), 0, 1) * (1. - 1e-5) + 1e-5
        mean, std = self.sde.distribution_params(x_0, t)
        epsilon = tf.random.normal( tf.shape(x_0))
        x_hat = mean + std * epsilon
        with tf.GradientTape() as tape:
            score = self([x_hat, t])
            score_loss = tf.square(score * std + epsilon)
            loss = tf.reduce_mean( 0.5 *tf.reduce_sum(score_loss, axis=[1,2,3]))

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.mean_loss.update_state(loss)
        return {m.name: m.result() for m in self.metrics}


def train_on_mnist():
    num_steps = 1000
    batch_size = 64
    images = load_mnist_dataset((70_000 // batch_size) *batch_size)
    model = SdeModel((28,28,1),1, sde.VpSDE(0.01, 20), num_steps, scale_by_simga=False  )

    adam =tf.keras.optimizers.Adam(learning_rate=0.0001)
    ema = tfa.optimizers.MovingAverage(adam)
    model.compile(optimizer=ema, loss="mse", run_eagerly=False)
    model.fit(x = images,epochs=10, batch_size=batch_size,verbose = 1)

    img = model.Euler_Maruyama_sampler(16, num_steps = num_steps)
    img_cliped = tf.clip_by_value(img, 0, 1)
    plot_images(img_cliped, 4, 4, 1, "gray", False)


def train_on_face(checkpoint_path = "./checkpoint/test", restore = False, load_path="./", init_epoch = None):

    def scheduler(epoch,lr):
        if epoch >= 0  and epoch <= 7:
            return 1e-3
        elif epoch >= 8 and epoch <= 16:
             return 1e-4
        else:
            return 1e-5

    def plot_batch(samples, frequency, checkpoint, model, num_steps):
        def plotter(epoch,logs = None):
            if (epoch+1)%frequency != 0:
                return
            #checkpoint.save()
            print("")
            images = model.pc_sampler(samples, num_steps)
            images = tf.clip_by_value(images, 0 , 1)
            plot_images(images, math.sqrt(samples),  math.sqrt(samples), epoch+1)
        return plotter

    dataPath = "C:\\Python\\PythonProjects\\DeepLearning\\character_data"# "C:\\Datasets\\Celeba_HQ"
    batch_size = 32
    samples = 16
    freq = 1
    initial_epoch = 0
    if init_epoch:
        initial_epoch = init_epoch
    num_steps = 1000

    filenames = [os.path.join(dataPath, i) for i in os.listdir(dataPath)]
    Gen_Train = tf.data.Dataset.from_tensor_slices((filenames))
    Gen_Train = Gen_Train.map(lambda x: process_path(x, x_res=128, y_res=128))
    Gen_Train = Gen_Train.shuffle(buffer_size=1024 * 4).batch(batch_size, True).prefetch(tf.data.AUTOTUNE).cache()


    model = SdeModel((128, 128, 3), 3, sde.VpSDE(0.1, 20, num_steps), 0.1, hiddens=[48, 96, 256, 512, 1024]) #new unet

    adam = tf.keras.optimizers.Adam(learning_rate=2e-5)
    ema = tfa.optimizers.MovingAverage(adam)
    model.compile(optimizer=ema, loss="mse", run_eagerly=False)

    checkpoint = tf.train.Checkpoint(model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=1)
    if restore:
        checkpoint.restore(load_path)
        img = model.pc_sampler(16, num_steps=num_steps)
        img_cliped = tf.clip_by_value(img, 0, 1)
        plot_images(img_cliped, 4, 4, 1, "rgb", False)

    epochs = 10
    model.fit(x=Gen_Train, epochs= epochs, batch_size=batch_size, verbose=1, initial_epoch = initial_epoch,
                           callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end= plot_batch(samples, freq, manager, model, num_steps)) ] )
    img = model.Euler_Maruyama_sampler(16, num_steps=num_steps)
    # img = img*0.5 + 0.5
    img_cliped = tf.clip_by_value(img, 0, 1)
    plot_images(img_cliped, 4, 4, 1, "rgb", False)
    model.summary()
    tf.keras.models.save_model(model, "saved_denoising_model")

if __name__ == "__main__":
    checkpoint_path = "./checkpoint"
    restore = False
    restore_path = None
    initial_epoch = 0


    xd = get_timestep_embedding(tf.random.uniform((100,1),1, 2, dtype=tf.int32), 64, 10000 )
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restore", action='store')
    parser.add_argument("-ie", "--initial_epoch", action='store', type = int)
    args = parser.parse_args()

    if args.restore:
        restore= True
        restore_path = args.restore
    if args.initial_epoch:
        initial_epoch = args.initial_epoch

    train_on_face(checkpoint_path, restore, restore_path, initial_epoch)
    #train_on_mnist()


