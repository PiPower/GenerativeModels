import tensorflow as tf

class SDE_example(tf.keras.layers.Layer):
    def __init__(self, sigma):
        super(SDE_example, self).__init__()
        self.sigma = tf.Variable(sigma, dtype= tf.float32, trainable=False)

    @tf.function
    def distribution_params(self, x, t):
        mean = x
        std_dev = tf.sqrt( (self.sigma ** (2 * t) - 1.) / (2. *tf.math.log(self.sigma)) )
        std_dev = tf.reshape(std_dev, (-1,1,1,1))
        return mean, std_dev

    @tf.function
    def sde(self, x, t):
        return x * (self.sigma**t)

    @tf.function
    def coefficients(self,x, t):
        return tf.zeros(tf.shape(x), dtype = tf.float32), self.sigma**t

    @tf.function
    def prior_sampling(self, shape):
        epsilon = tf.random.normal(shape)
        mean, stddev = self.distribution_params(tf.ones(1, dtype = tf.float32)*0, 1)
        return mean * epsilon * stddev

class VeSDE(tf.keras.layers.Layer):
    def __init__(self, sigma_min, sigma_max):
        super(VeSDE, self).__init__()
        self.sigma_min = tf.Variable(sigma_min, dtype= tf.float32, trainable=False)
        self.sigma_max = tf.Variable(sigma_max, dtype= tf.float32, trainable=False)

    @tf.function
    def distribution_params(self, x, t):
        mean = x
        std_dev = self.sigma_min * ( (self.sigma_max/self.sigma_min) **t )
        std_dev = tf.reshape(std_dev, (-1,1,1,1))
        return mean, std_dev

    @tf.function
    def coefficients(self,x, t):
        drift = tf.zeros(tf.shape(x), dtype = tf.float32)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * tf.sqrt(2 * tf.math.log( self.sigma_max/ self.sigma_min)  )
        return drift, diffusion

    @tf.function
    def prior_sampling(self, shape):
        epsilon = tf.random.normal(shape)
        mean, stddev = self.distribution_params(tf.zeros(shape), 1)
        return mean * epsilon * stddev


class VpSDE(tf.keras.layers.Layer):
    def __init__(self, beta_min, beta_max, N = 1000):
        super(VpSDE, self).__init__()
        self.beta_min= tf.Variable(beta_min, dtype= tf.float32, trainable=False)
        self.beta_max =  tf.Variable(beta_max, dtype=tf.float32, trainable=False)
        self.N = N
        self.discrete_betas = tf.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = tf.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    @tf.function
    def distribution_params(self,x, t):
        mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = tf.reshape(mean_coeff, (-1, 1, 1, 1))
        mean = tf.exp(mean_coeff) * x
        std = tf.sqrt(1.0 - tf.exp(2.0 * mean_coeff))
        return mean, std

    @tf.function
    def coefficients(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        beta_t = tf.reshape(beta_t, (-1,1,1,1))
        drift = -0.5 * beta_t * x
        diffusion = tf.sqrt(beta_t)
        return drift, diffusion

    @tf.function
    def prior_sampling(self, shape):
        return tf.random.normal(shape)

