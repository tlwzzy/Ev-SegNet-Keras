import tensorflow as tf
from keras.layers import Lambda, Input, Concatenate, Add
from keras.models import Model
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras import layers
from keras import regularizers


def resize_bilinear_(inputs, size, align_corners=False):
    return tf.image.resize_bilinear(inputs, size=size, align_corners=align_corners)


def upsampling(inputs, scale):
    a = inputs.shape[1]
    b = inputs.shape[2]
    # return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale],
    #                                 align_corners=True)
    # return tf.image.resize_bilinear(inputs, size=[a * scale, b * scale],
    #                                 align_corners=True)
    return Lambda(resize_bilinear_, arguments={'size': [a * scale, b * scale],
                                               'align_corners': True})(inputs)


def resize_bilinear__(x):
    inputs = x[0]
    input_to_copy = x[1]
    return tf.image.resize_bilinear(inputs, [input_to_copy.get_shape()[1].value,
                                             input_to_copy.get_shape()[2].value], align_corners=True)


def reshape_into(inputs, input_to_copy):
    # return tf.image.resize_bilinear(inputs, [input_to_copy.get_shape()[1].value,
    #                                          input_to_copy.get_shape()[2].value], align_corners=True)
    return Lambda(resize_bilinear__)([inputs, input_to_copy])


# convolution
def conv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Traspose convolution
def transposeConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=False):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0003),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0003),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


class Conv_BN(object):
    def __init__(self, filters, kernel_size, strides=1):
        # super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = conv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def __call__(self, inputs, activation=True):
        x = self.conv(inputs)
        x = self.bn(x)
        if activation:
            x = layers.LeakyReLU(alpha=0.3)(x)

        return x


class DepthwiseConv_BN(object):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        # super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def __call__(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = layers.LeakyReLU(alpha=0.3)(x)

        return x


class Transpose_Conv_BN(object):
    def __init__(self, filters, kernel_size, strides=1):
        # super(Transpose_Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = layers.LeakyReLU(alpha=0.3)(x)

        return x


class ShatheBlock(object):
    def __init__(self, filters, kernel_size, dilation_rate=1, bottleneck=2):
        # super(ShatheBlock, self).__init__()

        self.filters = filters * bottleneck
        self.kernel_size = kernel_size

        self.conv = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv1 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv2 = DepthwiseConv_BN(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.conv3 = Conv_BN(filters, kernel_size=1)

    def __call__(self, inputs):
        x = self.conv(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + inputs


def reduce_mean_(inputs, s, keep_dims):
    return tf.reduce_mean(inputs, s, keep_dims=keep_dims)


class ASPP_2(object):
    def __init__(self, filters, kernel_size):
        # super(ASPP_2, self).__init__()

        self.conv1 = DepthwiseConv_BN(filters, kernel_size=1, dilation_rate=1)
        self.conv2 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=4)
        self.conv3 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=8)
        self.conv4 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=16)
        self.conv6 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=(2, 8))
        self.conv7 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=(6, 3))
        self.conv8 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=(8, 2))
        self.conv9 = DepthwiseConv_BN(filters, kernel_size=kernel_size, dilation_rate=(3, 6))
        self.conv5 = Conv_BN(filters, kernel_size=1)

    def __call__(self, inputs, operation='concat'):
        feature_map_size = tf.shape(inputs)
        # image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
        image_features = Lambda(reduce_mean_, arguments={'s': [1, 2], 'keep_dims': True})(inputs)
        image_features = self.conv1(image_features)
        # image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
        image_features = Lambda(resize_bilinear_, arguments={'size': [feature_map_size[1], feature_map_size[2]]})(
            image_features)

        x1 = self.conv2(inputs)
        x2 = self.conv3(inputs)
        x3 = self.conv4(inputs)
        x4 = self.conv6(inputs)
        x5 = self.conv7(inputs)
        x4 = self.conv8(inputs) + x4
        x5 = self.conv9(inputs) + x5
        if 'concat' in operation:
            con = Concatenate(axis=3)([image_features, x1, x2, x3, x4, x5, inputs])
            # x = self.conv5(tf.concat((image_features, x1, x2, x3, x4, x5, inputs), axis=3))
            x = self.conv5(con)
        else:
            x = self.conv5(image_features + x1 + x2 + x3 + x5 + x4) + inputs

        return x


class EncoderAdaption(object):
    def __init__(self, filters, kernel_size, dilation_rate=1):
        # super(EncoderAdaption, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv1 = Conv_BN(filters, kernel_size=1)
        self.conv2 = ShatheBlock(filters, kernel_size=kernel_size, dilation_rate=dilation_rate)

    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class FeatureGeneration(object):
    def __init__(self, filters, kernel_size, dilation_rate=1, blocks=3):
        # super(FeatureGeneration, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv0 = Conv_BN(self.filters, kernel_size=1)
        self.blocks = []
        for n in range(blocks):
            self.blocks = self.blocks + [
                ShatheBlock(self.filters, kernel_size=kernel_size, dilation_rate=dilation_rate)]

    def __call__(self, inputs):

        x = self.conv0(inputs)
        for block in self.blocks:
            x = block(x)

        return x


def Xception(input_tensor, pooling=None):
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    outputs = []
    img_input = input_tensor

    channel_axis = -1

    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(img_input)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)
    outputs.append(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    # x = layers.add([x, residual])
    x = Add()([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)
    outputs.append(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    # x = layers.add([x, residual])
    x = Add()([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)
    outputs.append(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    # x = layers.add([x, residual])
    x = Add()([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        # x = layers.add([x, residual])
        x = Add()([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)
    outputs.append(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    # x = layers.add([x, residual])
    x = Add()([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    outputs.append(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)

    return x, outputs


def loss(y_true, y_pred):
    y_ = y_pred[0]
    aux_y_ = y_pred[1]
    loss = K.categorical_crossentropy(y_true, y_)
    loss_aux = K.categorical_crossentropy(y_true, aux_y_)
    # loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask)  # compute loss
    # loss_aux = tf.losses.softmax_cross_entropy(y, aux_y_, weights=mask)  # compute loss
    loss = 1 * loss + 0.8 * loss_aux
    return loss


def back_bone(outputs, num_class=1):
    # print("input", x.shape)
    # base_model = keras.applications.xception.Xception(include_top=False, input_shape=input_size, pooling='avg')(x)

    # Encoder
    adap_encoder_1 = EncoderAdaption(filters=128, kernel_size=3, dilation_rate=1)
    adap_encoder_2 = EncoderAdaption(filters=128, kernel_size=3, dilation_rate=1)
    adap_encoder_3 = EncoderAdaption(filters=128, kernel_size=3, dilation_rate=1)
    adap_encoder_4 = EncoderAdaption(filters=64, kernel_size=3, dilation_rate=1)
    adap_encoder_5 = EncoderAdaption(filters=32, kernel_size=3, dilation_rate=1)

    # Decoder
    decoder_conv_1 = FeatureGeneration(filters=128, kernel_size=3, dilation_rate=1, blocks=3)
    decoder_conv_2 = FeatureGeneration(filters=64, kernel_size=3, dilation_rate=1, blocks=3)
    decoder_conv_3 = FeatureGeneration(filters=32, kernel_size=3, dilation_rate=1, blocks=3)
    decoder_conv_4 = FeatureGeneration(filters=32, kernel_size=3, dilation_rate=1, blocks=1)
    aspp = ASPP_2(filters=32, kernel_size=3)

    # output
    conv_logits = conv(filters=num_class, kernel_size=1, strides=1, use_bias=True)

    # build the net
    # add activations to the ourputs of the model
    for i in range(len(outputs)):
        # print(outputs[i].shape)
        outputs[i] = layers.LeakyReLU(alpha=0.3)(outputs[i])

    # (outputs[0].shape)
    x = adap_encoder_1(outputs[0])

    x = upsampling(x, scale=2)
    x += reshape_into(adap_encoder_2(outputs[1]), x)  # 512
    x = decoder_conv_1(x)  # 256

    x = upsampling(x, scale=2)
    x += reshape_into(adap_encoder_3(outputs[2]), x)  # 256
    x = decoder_conv_2(x)  # 256
    # print("x1", x.shape)

    x = upsampling(x, scale=2)
    x += reshape_into(adap_encoder_4(outputs[3]), x)  # 128
    x = decoder_conv_3(x)  # 128
    # print("x1", x.shape)

    x = aspp(x, operation='sum')  # 128

    x = upsampling(x, scale=2)
    x += reshape_into(adap_encoder_5(outputs[4]), x)  # 64
    x = decoder_conv_4(x)  # 64
    # print("x1", x.shape)
    x = conv_logits(x)
    x = upsampling(x, scale=2)
    print("output", x.shape)
    aux_loss = False
    if aux_loss:
        return x, x
    else:
        return x


def segnet(input_size=(256, 256, 1), num_class=1, lr=0.001, momentum=0.9):
    from networks.unet import IoU_fun, mean_iou, IoU_loss_fun
    inputs = Input(input_size)
    base_model, outputs = Xception(inputs)
    # outputs = Lambda(Xception)(inputs)
    outputs.reverse()
    y_ = Lambda(back_bone, arguments={'num_class': num_class})(outputs)
    # y_, aux_y_ = back_bone(inputs, input_size=input_size, num_class=num_class)
    # y_ = back_bone(outputs, num_class)
    model = Model(input=inputs, output=y_)
    model.summary()
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss=IoU_loss_fun, metrics=['accuracy', mean_iou]) # categorical_crossentropy

    return model
