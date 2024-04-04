import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import refinenet
import tensorflow_addons as tfa
import os
import argparse


def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

def process_path(file_path, x_res = 80, y_res = 80):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [x_res, y_res])
    img = img / 255.0
    return img

def plot_images(images, rows, cols, index=0, mode = "rgb", save = True):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    if mode == "gray":
        images = tf.tile(images, [1,1,1,3])
    for y in range(rows):
        for x in range(cols):
            axs[y, x].imshow(images[y * rows + x, :, :, :])
    save_string = './test_images/score_model_{0}.png'.format(index)
    if save:
        plt.savefig(save_string)
    else:
        plt.show()
    plt.close()

def load_mnist_dataset(elements = None):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.concatenate([train_images, test_images], 0 )
    train_images = train_images.astype('float32') / 255.0
    train_images = np.reshape(train_images, newshape=(-1, 28, 28,1))
    if elements is not None and elements < train_images.shape[0]:
        return train_images[0:elements, :, : ]
    return train_images

class ScoreModel(tf.keras.Model):
    def __init__(self, shape, hiddens, start_var, q, L, epislon = 1e-5, T = 1000, embbeded_dim = 64):
        super(ScoreModel, self).__init__()
        self.epsilon = epislon
        self.target_shape = shape
        self.sgima_flatten = shape[0]*shape[1]*shape[2]
        refinenet.L = L # set refinenet global to our L
        self.model = refinenet.RefineNet(hiddens, tf.nn.elu)
        self.L = L
        self.T = T
        self.sigma = tf.range(0, L, dtype = tf.float32)
        self.sigma = start_var * tf.pow( q, self.sigma )
        self.embedding = tf.keras.layers.Embedding(self.L, embbeded_dim)
        self.pos_encoding = positional_encoding(self.L, embbeded_dim)
        self.time_steps = tf.range(0,self.L)
        self.time_steps = tf.reshape( self.time_steps, (-1, 1, 1))


        self.mean_loss = tf.keras.metrics.Mean(name="mean_loss")


    def sample(self, count):
        noise = tf.random.uniform( (count, ) + self.target_shape , 0, 1)
        normalization = tf.reshape( tf.square( tf.gather(self.sigma, (self.L-1))), (1))
        x_in = noise
        print()
        for i in range(self.L):
            sigma = tf.reshape( tf.gather(self.sigma, (i)), (1) )
            alpha_i = self.epsilon * tf.square(sigma) / normalization
            print("it is {0}th iteration".format(i))
            indicies = tf.ones( (count), dtype=tf.int32 ) * i
            for t in range(self.T):
                epsilon = tf.random.normal((count, ) + self.target_shape)
                x_next = x_in + alpha_i/2 * self.model([x_in, indicies] ) + tf.sqrt(alpha_i) * epsilon
                x_in = x_next
        return  x_in

    @tf.function
    def train_step(self, x):
        batch_size = tf.shape(x) * tf.convert_to_tensor([1,0,0,0])
        batch_size = tf.reduce_sum(batch_size)

        simga_idx = tf.random.uniform((batch_size,),0, self.L, dtype=tf.int32)
        sigmas = tf.reshape(tf.gather(self.sigma, simga_idx), (-1,1,1,1))
        epsilon = tf.random.normal(tf.shape(x))

        x_hat = x + epsilon * sigmas
        with tf.GradientTape() as tape:
            score = self.model([x_hat, simga_idx])
            score_loss = tf.square(score * sigmas + epsilon)
            loss = tf.reduce_mean( 0.5 *tf.reduce_sum(score_loss, axis=[1,2,3]))

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.mean_loss.update_state(loss)
        return {m.name: m.result() for m in self.metrics}



def train_on_face(checkpoint_path = "./checkpoint/test", restore = False, load_path="./", init_epoch = None):

    def scheduler(epoch,lr):
        if epoch >= 0  and epoch <= 7:
            return 1e-3
        elif epoch >= 8 and epoch <= 16:
             return 1e-4
        else:
            return 1e-5

    def plot_batch(samples, frequency, checkpoint, model):
        def plotter(epoch,logs = None):
            if (epoch+1)%frequency != 0:
                return
            checkpoint.save()
            generated_images = model.sample(samples)
            generated_images = tf.clip_by_value(generated_images, 0 , 1)
            plot_images(generated_images, 4,4,epoch +1, "rgb", True)
        return plotter

    dataPath = "C:\\Datasets\\Celeba_HQ"
    batch_size = 64
    samples = 16
    freq = 10
    initial_epoch = 0
    if init_epoch:
        initial_epoch = init_epoch

    filenames = [os.path.join(dataPath, i) for i in os.listdir(dataPath)]
    Gen_Train = tf.data.Dataset.from_tensor_slices((filenames))
    Gen_Train = Gen_Train.map(lambda x: process_path(x, x_res=56, y_res=56))
    Gen_Train = Gen_Train.shuffle(buffer_size=1024 * 4).batch(batch_size, True).prefetch(tf.data.AUTOTUNE).cache()

    model = ScoreModel( (56, 56, 3), 64, 30, 0.7395, 30, T = 100 )

    adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
    ema = tfa.optimizers.MovingAverage(adam)
    model.compile(optimizer=adam, loss="mse", run_eagerly=False)

    checkpoint = tf.train.Checkpoint(model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=1)
    if restore:
        checkpoint.restore(load_path)
    epochs = 1
    model.fit(x=Gen_Train, epochs= epochs, batch_size=batch_size, verbose=1, initial_epoch = initial_epoch,
                           callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end= plot_batch(samples, freq, manager, model)) ] )
    model.model.summary()


if __name__ == "__main__":
    checkpoint_path = "./checkpoint"
    restore = False
    restore_path = None
    initial_epoch = 0

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
