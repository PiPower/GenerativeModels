import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
L = 10

class DilatedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation=1, padding=1, strides=1):
        super(DilatedConv2D, self).__init__()

        self.padding = layers.ZeroPadding2D(padding)
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation, padding='valid')
        self.filters = filters
    @tf.function
    def call(self, inputs, **kwargs):
        x = self.padding(inputs)
        x = self.conv(x)
        return x


class ConditionalFullPreActivationBlock(layers.Layer):
    def __init__(self, activation, filters, kernel_size=3, dilation=1, padding=1, pooling=False):
        super(ConditionalFullPreActivationBlock, self).__init__()

        self.norm1 = ConditionalInstanceNormalizationPlusPlus2D()
        # FIXME: The number of filters in this convolution should be equal
        # to the input depth, instead of "filters"
        # The depth is increased only in the conv2
        self.conv1 = DilatedConv2D(filters, kernel_size, dilation, padding)
        self.norm2 = ConditionalInstanceNormalizationPlusPlus2D()
        self.conv2 = DilatedConv2D(filters, kernel_size, dilation, padding)
        self.pooling = pooling
        self.activation = activation

        self.increase_channels_skip = None

        self.filters = filters

    def build(self, input_shape):
        begin_filters = input_shape[0][-1]
        if begin_filters != self.filters:
            self.increase_channels_skip = layers.Conv2D(self.filters, kernel_size=1, padding='valid')
    @tf.function
    def call(self, inputs, **kwargs):
        skip_x, idx_sigmas = inputs
        x = self.norm1([skip_x, idx_sigmas])
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2([x, idx_sigmas])
        x = self.activation(x)
        x = self.conv2(x)

        if self.increase_channels_skip is not None:
            skip_x = self.increase_channels_skip(skip_x)

        if self.pooling:
            # FIXME: In the original code, there is a convolution before this pooling
            x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')
            skip_x = tf.nn.avg_pool2d(skip_x, ksize=2, strides=2, padding='SAME')

        return skip_x + x


class RCUBlock(ConditionalFullPreActivationBlock):
    def __init__(self, activation, filters, kernel_size=3, dilation=1):
        super(RCUBlock, self).__init__(activation, filters, kernel_size, dilation)


class ConditionalInstanceNormalizationPlusPlus2D(layers.Layer):
    def __init__(self):
        super(ConditionalInstanceNormalizationPlusPlus2D, self).__init__()
        self.L = L

        # FIXME: Here we initialize with ones instead of random normal around 1
        self.init_weights = 'ones'  # tf.random_normal_initializer(1, 0.02)
        self.init_bias = 'zeros'

    def build(self, input_shape):
        self.C = input_shape[0][-1]
        self.alpha = self.add_weight(name=self.name + '_alpha', shape=(self.L, 1, 1, self.C),
                                     initializer=self.init_weights)
        self.beta = self.add_weight(name=self.name + '_beta', shape=(self.L, 1, 1, self.C), initializer=self.init_bias)
        self.gamma = self.add_weight(name=self.name + '_gamma', shape=(self.L, 1, 1, self.C),
                                     initializer=self.init_weights)
    @tf.function
    def call(self, inputs, **kwargs):
        x, idx_sigmas = inputs
        mu, s = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        m, v = tf.nn.moments(mu, axes=[-1], keepdims=True)

        first = tf.gather(self.gamma, idx_sigmas) * (x - mu) / tf.sqrt(s + 1e-6)
        second = tf.gather(self.beta, idx_sigmas)
        third = tf.gather(self.alpha, idx_sigmas) * (mu - m) / tf.sqrt(v + 1e-6)

        z = first + second + third

        return z


class ConditionalChainedResidualPooling2D(layers.Layer):
    def __init__(self, n_blocks, activation, filters, kernel_size=3, pooling_size=5):
        super(ConditionalChainedResidualPooling2D, self).__init__()
        self.activation1 = activation
        self.n_blocks = n_blocks
        self.pooling_size = pooling_size
        for n in range(n_blocks):
            setattr(self, 'norm1{}'.format(n), ConditionalInstanceNormalizationPlusPlus2D())
            setattr(self, 'conv{}'.format(n), layers.Conv2D(filters, kernel_size, padding='same'))
    @tf.function
    def call(self, inputs, **kwargs):
        x, idx_sigmas = inputs
        x_residual = self.activation1(x)
        x = x_residual
        for n in range(self.n_blocks):
            norm1 = getattr(self, 'norm1{}'.format(n))
            conv = getattr(self, 'conv{}'.format(n))

            x = norm1([x, idx_sigmas])
            x = tf.nn.avg_pool2d(x, self.pooling_size, strides=1, padding='SAME')
            x = conv(x)
            x_residual += x
        return x_residual


class MultiResolutionFusion(layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(MultiResolutionFusion, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv2d_high = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.norm_high = ConditionalInstanceNormalizationPlusPlus2D()
        self.conv2d_low = None
        self.norm_low = None

    def build(self, input_shape):
        if len(input_shape[0]) == 2:
            self.norm_low = ConditionalInstanceNormalizationPlusPlus2D()
            self.conv2d_low = layers.Conv2D(self.filters, self.kernel_size, padding='same')
    @tf.function
    def call(self, inputs, **kwargs):
        idx_sigmas = inputs[1]
        assert len(inputs[0]) != 1, "Input in MRF of wrong size"

        if len(inputs[0]) == 2:
            high_input, low_input = inputs[0]

            low_input = self.norm_low([low_input, idx_sigmas])
            low_input = self.conv2d_low(low_input)
            low_input = tf.image.resize(low_input, high_input.shape[1:-1])
            high_input = self.norm_high([high_input, idx_sigmas])
            high_input = self.conv2d_high(high_input)

            return low_input + high_input


class RefineBlock(layers.Layer):
    def __init__(self, activation, filters, n_blocks_crp=2, n_blocks_begin_rcu=2, n_blocks_end_rcu=1, kernel_size=3, pooling_size=5):
        super(RefineBlock, self).__init__()

        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size

        self.n_blocks_begin_rcu = n_blocks_begin_rcu

        self.mrf = MultiResolutionFusion(filters, kernel_size)
        self.crp = ConditionalChainedResidualPooling2D(n_blocks_crp, activation, filters, kernel_size, pooling_size)
        self.n_blocks_end_rcu = n_blocks_end_rcu

    def build(self, input_shape):
        for n in range(self.n_blocks_begin_rcu):
            setattr(self, 'rcu_high{}'.format(n), RCUBlock(self.activation, self.filters, self.kernel_size))
            if len(input_shape) == 2:
                setattr(self, 'rcu_low{}'.format(n), RCUBlock(self.activation, self.filters, self.kernel_size))

        for n in range(self.n_blocks_end_rcu):
            setattr(self, 'end_rcu{}'.format(n), RCUBlock(self.activation, self.filters, self.kernel_size))
    @tf.function
    def call(self, inputs, **kwargs):
        idx_sigmas = inputs[1]
        if len(inputs[0]) == 1:
            high_input = inputs[0][0]

            for n in range(self.n_blocks_begin_rcu):
                rcu_high = getattr(self, 'rcu_high{}'.format(n))
                high_input = rcu_high([high_input, idx_sigmas])

            x = high_input

        elif len(inputs[0]) == 2:
            high_input, low_input = inputs[0]

            for n in range(self.n_blocks_begin_rcu):
                rcu_high = getattr(self, 'rcu_high{}'.format(n))
                rcu_low = getattr(self, 'rcu_low{}'.format(n))
                high_input = rcu_high([high_input, idx_sigmas])
                low_input = rcu_low([low_input, idx_sigmas])

            x = self.mrf([[high_input, low_input], idx_sigmas])

        x = self.crp([x, idx_sigmas])

        for n in range(self.n_blocks_end_rcu):
            end_rcu = getattr(self, 'end_rcu{}'.format(n))
            x = end_rcu([x, idx_sigmas])

        return x


class RefineNet(keras.Model):

    def __init__(self, filters, activation):
        super(RefineNet, self).__init__()
        self.in_shape = None

        self.increase_channels = layers.Conv2D(filters, kernel_size=3, padding='same')

        self.preact_1 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3)
        # FIXME: In this second preactivation block we used dilation=1, but it should have been 2
        self.preact_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, pooling=True)

        self.preact_3 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=2, padding=2)
        self.preact_4 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=4, padding=4)

        self.refine_block_1 = RefineBlock(activation, filters, n_blocks_crp=2, n_blocks_begin_rcu=2, n_blocks_end_rcu=3)
        self.refine_block_2 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_3 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_4 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)

        self.norm = ConditionalInstanceNormalizationPlusPlus2D()
        self.activation = activation
        self.decrease_channels = None

    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.in_shape = input_shape
        self.decrease_channels = layers.Conv2D(input_shape[0][-1], kernel_size=3, strides=1, padding='same')
    @tf.function
    def call(self, inputs, mask=None):
        x, idx_sigmas = inputs
        x = self.increase_channels(x)

        output_1 = self.preact_1([x, idx_sigmas])
        output_2 = self.preact_2([output_1, idx_sigmas])
        output_3 = self.preact_3([output_2, idx_sigmas])
        output_4 = self.preact_4([output_3, idx_sigmas])

        output_4 = self.refine_block_4([[output_4], idx_sigmas])
        output_3 = self.refine_block_3([[output_3, output_4], idx_sigmas])
        output_2 = self.refine_block_2([[output_2, output_3], idx_sigmas])
        output_1 = self.refine_block_1([[output_1, output_2], idx_sigmas])

        output = self.norm([output_1, idx_sigmas])
        output = self.activation(output)
        output = self.decrease_channels(output)

        return output


class RefineNetTwoResidual(keras.Model):

    def __init__(self, filters, activation):
        super(RefineNetTwoResidual, self).__init__()
        self.in_shape = None

        self.increase_channels = layers.Conv2D(filters, kernel_size=3, padding='same')

        self.preact_1_1 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3)
        self.preact_1_2 = ConditionalFullPreActivationBlock(activation, filters, kernel_size=3)
        # FIXME: In this second preactivation block we used dilation=1, but it should have been 2
        self.preact_2_1 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3)
        self.preact_2_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, pooling=True)

        self.preact_3_1 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=2,
                                                            padding=2)
        self.preact_3_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=2,
                                                            padding=2)
        self.preact_4_1 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=4,
                                                            padding=4)
        self.preact_4_2 = ConditionalFullPreActivationBlock(activation, filters * 2, kernel_size=3, dilation=4,
                                                            padding=4)

        self.refine_block_1 = RefineBlock(activation, filters, n_blocks_crp=2, n_blocks_begin_rcu=2, n_blocks_end_rcu=3)
        self.refine_block_2 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_3 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)
        self.refine_block_4 = RefineBlock(activation, filters * 2, n_blocks_crp=2, n_blocks_begin_rcu=2)

        self.norm = ConditionalInstanceNormalizationPlusPlus2D()
        self.activation = activation
        self.decrease_channels = None

    def build(self, input_shape):
        # Here we get the depth of the image that is passed to the model at the start, i.e. 1 for MNIST.
        self.in_shape = input_shape
        self.decrease_channels = layers.Conv2D(input_shape[0][-1], kernel_size=3, strides=1, padding='same')

    @tf.function
    def call(self, inputs, mask=None):
        x, idx_sigmas = inputs
        x = self.increase_channels(x)

        output_1 = self.preact_1_1([x, idx_sigmas])
        output_1 = self.preact_1_2([output_1, idx_sigmas])
        output_2 = self.preact_2_1([output_1, idx_sigmas])
        output_2 = self.preact_2_2([output_2, idx_sigmas])
        output_3 = self.preact_3_1([output_2, idx_sigmas])
        output_3 = self.preact_3_2([output_3, idx_sigmas])
        output_4 = self.preact_4_1([output_3, idx_sigmas])
        output_4 = self.preact_4_2([output_4, idx_sigmas])

        output_4 = self.refine_block_4([[output_4], idx_sigmas])
        output_3 = self.refine_block_3([[output_3, output_4], idx_sigmas])
        output_2 = self.refine_block_2([[output_2, output_3], idx_sigmas])
        output_1 = self.refine_block_1([[output_1, output_2], idx_sigmas])

        output = self.norm([output_1, idx_sigmas])
        output = self.activation(output)
        output = self.decrease_channels(output)

        return output


