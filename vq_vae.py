import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np
import os
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

"""
based on
https://keras.io/examples/generative/vq_vae/
""" 
def process_path(file_path, x_res=80, y_res=80):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [x_res, y_res])
    img = img / 127.5 - 1
    return img


def plot_images(images, rows, cols, index=0, mode = "rgb", save = True):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    if mode == "gray":
        images = tf.tile(images, [1,1,1,3])
    for y in range(rows):
        for x in range(cols):
            axs[y, x].imshow(images[y * rows + x, :, :, :])
    save_string = './test_images/test_batch_hierarchical_{0}.png'.format(index)
    if save:
        plt.savefig(save_string)
    else:
        plt.show()
    plt.close()

def create_mnist_dataset():
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    total_dataset = np.concatenate( [train_images,test_images] , axis=0)
    total_dataset  = total_dataset/127.5 - 1
    total_dataset = np.expand_dims(total_dataset, axis=-1)
    return  total_dataset

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

def decoder_front(noise_shape, hidden_dim, out_channels, depth):
    input_dec = tf.keras.layers.Input(noise_shape)
    x = input_dec
    for i in range(len(hidden_dim)):
        x = ResblockUp(x, hidden_dim[i], i, depth)(x)

    out_dec = tf.keras.layers.Conv2D(out_channels, (3, 3), padding='same')(x)
    return  tf.keras.models.Model(inputs=input_dec, outputs=out_dec)

def encoder_front(start_res, hidden_dim, latent_dim, depth):
    input = tf.keras.layers.Input(start_res)
    x = input
    for i in range(len(hidden_dim)):
        x = ResblockDown(x, hidden_dim[i], i, depth)(x)

    target = layers.Conv2D(latent_dim, 3, padding="same")(x)
    return tf.keras.models.Model(inputs=input, outputs=[target])

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = tf.reduce_mean(tf.reduce_sum((tf.stop_gradient(quantized) - x) ** 2, axis=-1) )
        codebook_loss =tf.reduce_mean(  tf.reduce_sum((quantized - tf.stop_gradient(x)) ** 2, axis=-1) )
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

def get_vqvae(enc_hidden_dim, dec_hidden_dim, input_shape, noise_shape ,latent_dim=16, num_embeddings=64, depth = 1):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder =  encoder_front(input_shape, enc_hidden_dim, latent_dim, depth)
    decoder = decoder_front(noise_shape, dec_hidden_dim, input_shape[2], depth)
    inputs = keras.Input(shape=input_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


class VQVAETrainer(keras.models.Model):
    def __init__(self,target_resolution ,train_variance, enc_hidden_dim, dec_hidden_dim, latent_dim=32, num_embeddings=128, depth = 1,  **kwargs):
        super().__init__(**kwargs)
        assert target_resolution[0]%2**len(enc_hidden_dim) == 0
        assert target_resolution[1] % 2**len(enc_hidden_dim) == 0

        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.target_res = target_resolution
        self.depth = depth
        self.noise_shape = (target_resolution[0]//2**len(enc_hidden_dim),target_resolution[1]//2**len(enc_hidden_dim), latent_dim)
        self.vqvae = get_vqvae(enc_hidden_dim, dec_hidden_dim,target_resolution, self.noise_shape, self.latent_dim, self.num_embeddings, depth)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]
    def sample(self, count):
        #noise = tf.random.uniform((count, self.noise_shape[0], self.noise_shape[1], self.noise_shape[2]),
        #0, self.num_embeddings, dtype=tf.int32  )
        x_in = tf.zeros(shape=(count, ) + self.target_res)
        x = self.vqvae(x_in)
        return x

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_sum((x - reconstructions) ** 2, axis=[1,2,3])/self.train_variance
            )
            total_loss = tf.reduce_mean(reconstruction_loss) + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }


def train_on_face(checkpoint_path="./checkpoint/test", restore=False, load_path="./", init_epoch=None):
    def scheduler(epoch,lr):
        if epoch >= 0  and epoch <= 20:
            return 1e-3
        elif epoch >= 20 and epoch <= 40:
             return 1e-4
        else:
            return 1e-5

    def epoch_end_ops(frequency, checkpoint, model, test_dataset):
        def plotter(epoch,logs = None):
            if (epoch+1)%frequency != 0:
                return
            checkpoint.save()
            print("\n")

            images = model.vqvae(np.array(test_dataset))
            images = tf.unstack(images) + test_dataset
            images = [0.5 * img + 0.5 for img in images]
            to_plot = tf.convert_to_tensor(images)
            plot_images(to_plot, 4, 4, epoch)

        return plotter


    batch_size = 64
    initial_epoch = 0
    if init_epoch:
        initial_epoch = init_epoch

    dataPath = "C:\\Datasets\\Celeba_HQ"
    filenames = [os.path.join(dataPath, i) for i in os.listdir(dataPath)]
    test_size = 8
    test_batch = filenames[:test_size]
    test_images = [process_path(file_path, 64,64) for file_path in test_batch]
    Gen_Train = tf.data.Dataset.from_tensor_slices((filenames[test_size:]))
    Gen_Train = Gen_Train.map(lambda x: process_path(x, x_res=64, y_res=64))
    Gen_Train = Gen_Train.shuffle(buffer_size=1024 * 4).batch(batch_size, True).prefetch(tf.data.AUTOTUNE).cache()

    model = VQVAETrainer( (64,64,3), 1, [64, 128, 256, 512], [512, 256, 128, 64],depth = 3, latent_dim=128, num_embeddings = 512)
    adam = tf.keras.optimizers.Adam()
    model.compile( optimizer=adam, loss= "MSE", run_eagerly= False)
    checkpoint = tf.train.Checkpoint(model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=1)
    if restore:
        checkpoint.restore(load_path)

    epochs = 1
    #Gen_Train = create_mnist_dataset()
    model.fit(x=Gen_Train, epochs=epochs, batch_size=batch_size, verbose=1, initial_epoch=initial_epoch,
              callbacks = [
                  tf.keras.callbacks.LearningRateScheduler(scheduler),
                  tf.keras.callbacks.LambdaCallback(on_epoch_end=epoch_end_ops(20, manager, model, test_images))
              ])
    model.vqvae.summary()

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
