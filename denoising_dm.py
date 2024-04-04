import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import math
import os
import argparse
import  new_test_model
from PIL import Image

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
    img = img / 127.5 - 1
    return img

def plot_images(images, rows, cols, index = 0):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    for y in range(rows):
        for x in range(cols):
            axs[y, x].imshow(images[y*rows + x, :, :, :])
    save_string = './test_images/test_batch_{0}.png'.format(index)
    plt.savefig(save_string)
    plt.close()

def display_transition(image_sequece, transition_count, index, mode = "gray", epoch = None):

    fig = plt.figure(figsize=(30, 30))
    num_images = int(tf.shape(image_sequece[0])[0])
    step_size = math.ceil(len(image_sequece) / transition_count)
    curr = 0

    fig, axs = plt.subplots(num_images, transition_count, figsize=(15, 15))
    while curr < len(image_sequece):
        current_batch = image_sequece[curr]
        max_val = np.amax(np.abs(current_batch))
        current_batch = current_batch / max_val
        current_batch = 0.5 * current_batch + 0.5
        for j in range(num_images):
            if mode == "gray":
                axs[j, curr//step_size ].imshow(current_batch[j,:,:,0], cmap='gray')
            if mode == "rgb":
                axs[j, curr // step_size].imshow(current_batch[j, :, :, :])

            plt.axis('off')

        curr = curr + step_size

    current_batch = image_sequece[-1]
    # image proportional normalization if value are greater than 1
    max_val = np.amax(tf.abs(current_batch))
    current_batch = current_batch/max_val
    current_batch = 0.5 * current_batch + 0.5
    for j in range(num_images):
        axs[j, transition_count - 1].imshow(current_batch[j, :, :], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    save_string = './test_images/test_transition_{0}.png'.format(index)
    if epoch is not None:
        save_string = './test_images/test_transition_ep_{1}_it_{0}.png'.format(index, epoch)

    plt.savefig(save_string)
    plt.close()

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-4, end=0.02):
    if schedule == 'linear':
        betas = tf.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = tf.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = tf.linspace(-6, 6, n_timesteps)
        betas = tf.sigmoid(betas) * (end - start) + start
    return betas


def load_mnist_dataset(elements = None):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.concatenate([train_images, test_images], 0 )
    train_images = train_images.astype('float32') / 127.5 - 1
    train_images = np.reshape(train_images, newshape=(-1, 28, 28,1))
    if elements is not None and elements < train_images.shape[0]:
        return train_images[0:elements, :, : ]
    return train_images

@tf.function
def extract_values_from(source, indicies, indicies_size, y_size):
    t = tf.reshape(indicies, (-1, 1))
    t = tf.tile(t, [1, indicies_size])

    mask = tf.range(0,indicies_size)
    mask = tf.expand_dims(mask, 0)
    mask = tf.tile(mask, [y_size, 1])

    source_vec = tf.cast(tf.reshape(source, (-1,1)), dtype = 'float32' )
    mask = tf.cast(tf.math.equal(mask, t), dtype='float32')
    mask = tf.matmul(mask,source_vec)
    return  mask

def non_local_block(filters, height, width):
    assert filters%2 ==0
    x = layers.Input( shape = (height, width, filters) )

    g = layers.Conv2D(filters/2, (1,1), padding = 'same') (x)
    theta = layers.Conv2D(filters/2, (1,1), padding = 'same')(x)
    phi = layers.Conv2D(filters/2, (1,1), padding = 'same')(x)

    phi = layers.MaxPool2D()(phi)
    g = layers.MaxPool2D()(g)

    theta = tf.reshape(theta, (-1,height*width, filters // 2))
    phi = tf.reshape(phi, (-1, filters // 2, height * width//4,))
    g = tf.reshape(g, (-1, height*width//4, filters//2))

    y = tf.nn.softmax( tf.matmul(theta, phi), 1 )
    z = tf.matmul(y, g)
    z = tf.reshape(z, (-1, height, width, filters//2))


    z = layers.Conv2D(filters, (1, 1),)(z)
    z = z + x

    return tf.keras.models.Model(x,z)

def double_conv_block(x, n_filters, dropout_rate):

    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    return x

def downsample_block(x,t, n_filters, dropout_rate):
    x = x + t
    #x = x + t
    f = double_conv_block(x, n_filters, dropout_rate)
    p =  layers.Conv2D(n_filters, 3, 2,  padding="same")(f)

    return f, p

def upsample_block(x, t,skip, n_filters, dropout_rate):
    # upsample
    x = x + t
    #x = x + t
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.Dropout(dropout_rate)(x)
    # dropout
    x = x + skip
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters, dropout_rate)

    return x

def build_unet_model(shape,output_channels, num_steps, hidden_dims, drop_out_rate=0.1, att_block = -1, embedd_dim  = 64):
    # inputs
    num_blocks = len(hidden_dims)
    embedding_input = layers.Input(shape=(embedd_dim))
    inputs = layers.Input(shape=shape)

    x = inputs
    skips = []
    for i in range(num_blocks):
        t_in = layers.Dense( shape[0] // 2 ** i * shape[1] // 2 ** i * 1, activation = 'relu' )(embedding_input)
        t_in = layers.Reshape((shape[0] // 2 ** i, shape[1] // 2 ** i, 1))(t_in)
        skip_conn, x = downsample_block(x, t_in,  hidden_dims[i], drop_out_rate  )
        if i == att_block:
            x_out = non_local_block(hidden_dims[i], shape[0]//2**(i+1), shape[1]//2**(i+1) )(x)
            x = x_out + x
        #x = layers.BatchNormalization()(x)
        skips.append(skip_conn)


    for i in range(num_blocks-1 ,-1,-1):
        t_in =layers.Dense(  shape[0] // 2 ** (i+1) * shape[1] // 2 ** (i+1) * 1, activation = 'relu' )(embedding_input)
        t_in = layers.Reshape((shape[0] // 2 ** (i+1), shape[1] // 2 ** (i+1), 1))(t_in)
        x = upsample_block(x, t_in, skips[i],  hidden_dims[i], drop_out_rate)
        if (i + 1) == num_blocks - att_block:
            denominator = 2 **(i)
            x_out = non_local_block(hidden_dims[i],shape[0]//denominator, shape[1]//denominator )(x)
            x = x_out + x
        #x = layers.BatchNormalization()(x)
    outputs = layers.Conv2D(output_channels, 3, padding="same")(x)
    unet_model = tf.keras.Model(inputs=[embedding_input, inputs], outputs=outputs, name="U-Net")

    return unet_model


class DiffusionModel(tf.keras.Model):
    def __init__(self, shape, channels, num_steps=30, batch_size=128, drop_out_rate=0.1, att_block = 0, hiddens = [48,86], unet_old = True, embedd_dim = 64):
        super(DiffusionModel, self).__init__()
        if unet_old:
            self.model = build_unet_model(shape ,channels, num_steps,hiddens , drop_out_rate,  att_block, embedd_dim = embedd_dim)
        else:
            self.model = new_test_model.get_network(shape[0], hiddens,2, embedd_dim)
        self.hiddens = hiddens
        self.embedd_dim = embedd_dim
        self.unet_old = unet_old
        self.num_steps = num_steps
        self.att_block = att_block
        self.drop_out_rate = drop_out_rate
        self.channels = channels
        self.shape = shape
        self.batch_size = batch_size
        betas = make_beta_schedule(schedule='linear', n_timesteps=num_steps + 1, start=1e-4, end=0.02)
        self.betas = tf.cast(betas, dtype="float32")
        self.att_block = att_block
        self.alphas = 1 - betas
        self.pos_encoding = positional_encoding( self.num_steps, embedd_dim)
        alphas_reduced = tf.unstack(self.alphas)
        del alphas_reduced[-1]
        alphas_reduced = tf.stack(alphas_reduced)
        alphas_reduced = tf.cast(alphas_reduced, dtype="float32")
        self.alphas_prod = tf.math.cumprod(self.alphas, 0)
        self.alphas_prod_p = tf.concat([tf.ones((1), dtype="float32"), alphas_reduced], 0)
        self.alphas_bar_sqrt = tf.cast(tf.sqrt(self.alphas_prod), dtype='float32')
        self.one_minus_alphas_bar_sqrt = tf.cast(tf.sqrt(1 - self.alphas_prod), dtype='float32')


    def get_config(self):
        return {
                "layer_config":{
                    "shape": self.shape,
                    "channels": self.channels,
                    "num_steps":self.num_steps,
                    "batch_size":self.batch_size,
                    "drop_out_rate": self.drop_out_rate,
                    "att_block":self.att_block,
                    "hiddens":self.hiddens,
                    "unet_old":self.unet_old,
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
        return self.model(inputs, training, mask)

    def sample_reversed(self, size = 4, freq = None):
        cur_x = tf.random.normal((size,) + self.shape)
        x_seq = [cur_x]
        reversed_proc = tf.ones(( size,1,1,1 ))
        final_step =  tf.zeros(( size,1,1,1 ))
        mask_in = reversed_proc
        for i in reversed(range(self.num_steps)):
            if i%100 == 0 :
                print("iter nr: {}".format(i))
                if i == 0 :
                    mask_in = final_step

            cur_x = self.p_sample(cur_x, tf.convert_to_tensor([i for _ in range(size)]), mask_in )
            x_seq.append(cur_x.numpy())
        return x_seq

    @tf.function
    def p_sample(self, x, t, mask):
        # Factor to the model output
        alpha = 1 - tf.reshape(tf.gather( self.betas, t ), (-1, 1, 1, 1))
        gamma = tf.reshape(tf.gather( self.one_minus_alphas_bar_sqrt, t ),  (-1, 1, 1, 1))

        eps_factor = ((1 - alpha) / gamma)
        # Model output
        time_encoding = tf.gather(  self.pos_encoding, t, axis=0 )
        eps_theta = self([time_encoding, x])
        # Final values
        mean = (1 / tf.sqrt(alpha)) * (x - (eps_factor * eps_theta))
        # Generate z
        z = tf.random.normal(tf.shape(x))
        sigma_t = tf.sqrt(tf.reshape(tf.gather( self.betas, t ), (-1, 1, 1, 1)))

        sample = mean + mask * sigma_t * z
        return sample

    @tf.function
    def train_step(self, x_0):
        batch_size = self.batch_size

        t = tf.random.uniform(  (batch_size,), 0, self.num_steps, dtype=tf.int32)
        time_encoding = tf.gather(  self.pos_encoding, t, axis=0 )
        a = tf.reshape(tf.gather( self.alphas_bar_sqrt, t ), (-1,1,1,1))
        gammas = tf.reshape( tf.gather(self.one_minus_alphas_bar_sqrt, t), (-1,1,1,1))
        noise = tf.random.normal(tf.shape(x_0))

        x = x_0 * a + noise * gammas

        with tf.GradientTape() as tape:
            y_pred = self([time_encoding, x], training=True)  # Forward pass
            loss = self.compiled_loss(noise, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # Return a dict mapping metric names to current value

        return {m.name: m.result() for m in self.metrics}

@tf.function
def DiffusionLoss(noise, prediction ):
    out = tf.square(noise - prediction)
    out = tf.math.reduce_mean(out)
    return out

def train_on_mnist():
    samples = 10
    num_steps = 500
    batch_size = 128
    images = load_mnist_dataset((70_000 // batch_size) *batch_size)
    #images = load_mnist_dataset(9*batch_size)
    #images = 2*images - 1

    model = DiffusionModel((28,28,1),1, num_steps, batch_size )

    adam =tf.keras.optimizers.Adam(learning_rate=0.005)
    ema = tfa.optimizers.MovingAverage(adam)
    model.compile(optimizer=ema, loss=DiffusionLoss)
    model.fit(x = images,epochs=20, batch_size=batch_size,verbose = 1)

    for i in range(samples):
        img_seq = model.sample_reversed()
        display_transition(img_seq, 5, i)

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
            print("\n")

            img_seq = model.sample_reversed(4*samples)
            for j in range(0, samples):
                img_seq_loop = [tensor[j * 4: (j + 1) * 4, :, :, :] for tensor in img_seq]
                display_transition(img_seq_loop, 5, j, "rgb",epoch+1)
                plt.close()
            del img_seq

        return plotter

    dataPath = "C:\\Datasets\\Celeba_HQ"
    batch_size = 40
    samples = 10
    freq = 10
    initial_epoch = 0
    if init_epoch:
        initial_epoch = init_epoch
    num_steps = 1000

    filenames = [os.path.join(dataPath, i) for i in os.listdir(dataPath)]
    Gen_Train = tf.data.Dataset.from_tensor_slices((filenames))
    Gen_Train = Gen_Train.map(lambda x: process_path(x, x_res=128, y_res=128))
    Gen_Train = Gen_Train.shuffle(buffer_size=1024 * 4).batch(batch_size, True).prefetch(tf.data.AUTOTUNE).cache()
    model = DiffusionModel((128, 128, 3), 3, num_steps, batch_size, hiddens=[48, 96, 256, 512, 1024], att_block=-2, unet_old = False) #new unet

    adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
    ema = tfa.optimizers.MovingAverage(adam)
    model.compile(optimizer=ema, loss=DiffusionLoss)

    checkpoint = tf.train.Checkpoint(model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=1)
    if restore:
        #checkpoint.restore(load_path)
        model = tf.saved_model.load(load_path)
        images = model.sample_reversed(16)[-1]
        images = images * 0.5 + 0.5
        plot_images(images,4, 4 )


    epochs = 100
    model.fit(x=Gen_Train, epochs= epochs, batch_size=batch_size, verbose=1, initial_epoch = initial_epoch,
                           callbacks=[ tf.keras.callbacks.LearningRateScheduler(scheduler),
                          tf.keras.callbacks.LambdaCallback(on_epoch_end= plot_batch(samples, freq, manager, model)) ] )
    model.summary()
    tf.keras.models.save_model(model, "saved_denoising_model")

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

