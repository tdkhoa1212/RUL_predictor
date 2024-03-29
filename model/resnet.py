import tensorflow as tf
from model.residual_block import make_basic_block_layer, make_bottleneck_layer
from tensorflow.keras.layers import Dropout

class ResNetTypeI(tf.keras.Model):
    def __init__(self, opt, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.Dropout = Dropout(0.2)
        
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training) 
        # x = self.avgpool(x)
        return x


class ResNetTypeII(tf.keras.Model):
    def __init__(self, opt, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.expand_dims = tf.expand_dims
        self.fc = tf.keras.layers.Dense(units=1024, activation='relu')
        self.Dropout = Dropout(0.2)
        
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        # x = self.Dropout(x, training=training)
        x = self.layer2(x, training=training)
        # x = self.Dropout(x, training=training)
        x = self.layer3(x, training=training)
        # x = self.Dropout(x, training=training)
        x = self.layer4(x, training=training)
        # x = self.Dropout(x, training=training)
        # x = self.avgpool(x)
        return x


def resnet_18(opt):
    return ResNetTypeI(opt, layer_params=[2, 2, 2, 2])


def resnet_34(opt):
    return ResNetTypeI(opt, layer_params=[3, 4, 6, 3])


def resnet_50(opt):
    return ResNetTypeII(opt, layer_params=[3, 4, 6, 3])


def resnet_101(opt):
    return ResNetTypeII(opt, layer_params=[3, 4, 23, 3])


def resnet_152(opt):
    return ResNetTypeII(opt, layer_params=[3, 8, 36, 3])
