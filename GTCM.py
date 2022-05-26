import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras.layers
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.layers import LeakyReLU,PReLU,add,ReLU
from tensorflow.keras.layers import Conv1D, SpatialDropout1D,Dropout
from tensorflow.keras.layers import DepthwiseConv2D,multiply
from tensorflow.keras.layers import Convolution1D, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import  Input
from tensorflow.keras.models import Model
from typing import List, Tuple
from tensorflow.keras.optimizers import RMSprop

def mul(x):
    return tf.multiply(x[0],x[1])

def makemean(x):
    shape = (K.int_shape(x)[1],K.int_shape(x)[2])
    meantensor = np.ones(shape,dtype=np.float32)
    meantensor = meantensor/3
    meantensor = K.expand_dims(meantensor,0)
    meantensor = tf.convert_to_tensor(meantensor)
    return meantensor

def residual_block(x, s, i, activation, nb_filters, kernel_size, dropout_rate=0, name=''):

    original_x = x
    #The First Level
    #1.1
    conv_1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal',
                  name=name + '_dilated_conv1.1_dilation%d_stack%d' % (i, s))(x)
    conv_1_1 =  Activation('relu')(conv_1_1)

    conv_s1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal',
                  name=name + '_dilated_convs1.1_dilation%d_stack%d' % (i, s))(x)
    conv_s1_1 =  Activation('relu')(conv_s1_1)
    conv_s1_1 = Lambda(sigmoid)(conv_s1_1)
    output_1_1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_1_1,conv_s1_1])
    #1.2
    conv_1_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal',
                  name=name + '_dilated_conv1.2_dilation%d_stack%d' % (i, s))(x)
    conv_1_2 =  Activation('relu')(conv_1_2)

    conv_s1_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal',
                  name=name + '_dilated_convs1.2_dilation%d_stack%d' % (i, s))(x)
    conv_s1_2 =  Activation('relu')(conv_s1_2)
    conv_s1_2 = Lambda(sigmoid)(conv_s1_2)
    output_1_2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_1_2,conv_s1_2])
    #1.3
    conv_1_3 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal',
                  name=name + '_dilated_conv1.3_dilation%d_stack%d' % (i, s))(x)
    conv_1_3 =  Activation('relu')(conv_1_3)

    conv_s1_3 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal',
                  name=name + '_dilated_convs1.3_dilation%d_stack%d' % (i, s))(x)
    conv_s1_3 =  Activation('relu')(conv_s1_3)
    conv_s1_3 = Lambda(sigmoid)(conv_s1_3)
    output_1_3 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_1_3,conv_s1_3])

    output_1 = add([output_1_1 , output_1_2 , output_1_3])
    templayer = Lambda(makemean)(output_1)
    output_1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([output_1,templayer])


    #The Second Level
    #2.1
    conv_2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i*2, padding='causal',
                  name=name + '_dilated_conv2.1_dilation%d_stack%d' % (i, s))(output_1)
    conv_2_1 = Activation('relu')(conv_2_1)

    conv_s2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i*2, padding='causal',
                  name=name + '_dilated_convs2.1_dilation%d_stack%d' % (i, s))(output_1)
    conv_s2_1 = Activation('relu')(conv_s2_1)
    conv_s2_1 = Lambda(sigmoid)(conv_s2_1)

    output_2_1 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_2_1,conv_s2_1])
    #2.2
    conv_2_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i*2, padding='causal',
                  name=name + '_dilated_conv2.2_dilation%d_stack%d' % (i, s))(output_1)
    conv_2_2 = Activation('relu')(conv_2_2)

    conv_s2_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i*2, padding='causal',
                  name=name + '_dilated_convs2.2_dilation%d_stack%d' % (i, s))(output_1)
    conv_s2_2 = Activation('relu')(conv_s2_2)
    conv_s2_2 = Lambda(sigmoid)(conv_s2_2)

    output_2_2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_2_2,conv_s2_2])
    #2.3
    conv_2_3 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i*2, padding='causal',
                  name=name + '_dilated_conv2.3_dilation%d_stack%d' % (i, s))(output_1)
    conv_2_3 = Activation('relu')(conv_2_3)

    conv_s2_3 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i*2, padding='causal',
                  name=name + '_dilated_convs2.3_dilation%d_stack%d' % (i, s))(output_1)
    conv_s2_3 = Activation('relu')(conv_s2_3)
    conv_s2_3 = Lambda(sigmoid)(conv_s2_3)

    output_2_3 = Lambda(lambda x: tf.multiply(x[0], x[1]))([conv_2_3,conv_s2_3])

    output_2 = add([output_2_1 , output_2_2 , output_2_3])
    templayer1 = Lambda(makemean)(output_2)
    output_2 = Lambda(lambda x: tf.multiply(x[0], x[1]))([output_2,templayer1])
    if original_x.shape[-1] != output_2.shape[-1]:
        original_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(original_x)
    res_x = tensorflow.keras.layers.add([original_x, output_2])
    return res_x, output_2


class GTCM:
    """Creates a Gated Temporal Convolution Module.
        Args:
            input_layer: 张量形状 (batch_size, timesteps, input_dim).
            nb_filters: 在卷积层中使用的filters数。
            kernel_size: 在每个卷积层中使用的kernel大小。
            dilations: 膨胀列表。 示例为： [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : 使用的残差块的堆栈数目。The number of stacks of residual blocks to use.
            activation: 使用的一些激活函数 (norm_relu, wavenet, relu...).
            use_skip_connections: 使用布尔值类型. 如果要添加跳层连接，就设置为True。If we want to add skip connections from input to each residual block.
            return_sequences: 使用布尔值类型.是否返回输出序列的最后输出，或者整个序列。 Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: 位于区间[0,1]的浮点数，用于对输出单元的dropout。Fraction of the input units to drop.
            name: 对模型的命名，Name of the model. Useful when having multiple TCN.
        Returns:
            A GTCM layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation='relu',
                 use_skip_connections=True,
                 dropout_rate=0.1,
                 return_sequences=True,
                 name='GTCM'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        if not isinstance(nb_filters, int):
            raise Exception()

    def __call__(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2,4, 8,16, 32,64,128,256]
        x = inputs
        x = Convolution1D(self.nb_filters, 1, padding='causal')(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x, skip_out = residual_block(x, s, i, self.activation,  
                                                        self.nb_filters,
                                                        self.kernel_size, 
                                                        self.dropout_rate,  
                                                        name=self.name)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = tensorflow.keras.layers.add(skip_connections)
        x = LeakyReLU(alpha=0.05)(x)
        if not self.return_sequences:
            output_slice_index = -1
            x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
        return x

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1],1)