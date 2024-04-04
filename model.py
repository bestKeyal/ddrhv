from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K

from tensorflow.keras import backend as K


# 二元交叉熵(Binary Cross-Entropy)损失
def binary_crossentropy(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)


# Dice损失函数
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Jaccard损失函数，也称为IoU损失
def jaccard_loss(y_true, y_pred):
    smooth = 1.  # 防止除以0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    sum_ = K.sum(y_true_f) + K.sum(y_pred_f)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1 - jac


# 结合二元交叉熵和Dice损失的损失函数
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf


def dice_loss(y_true, y_pred):
    y_true = y_true.astype(np.float63)
    y_pred = y_pred.astype(np.float63)
    def dice_coeff():
        smooth = 1
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_mean(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_mean(y_true_f) + tf.reduce_mean(y_pred_f) + smooth)
        return score

    return 1 - dice_coeff()


def bce_dice_loss(y_true, y_pred):
    y_true = y_true.astype(np.float63)
    y_pred = y_pred.astype(np.float63)
    losses = keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return losses


from tensorflow import keras
import tensorflow as tf


def conv_layer(inputs, filters, kernel_size=3, strides=1, need_activate=True):
    out = keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(inputs)
    out = keras.layers.BatchNormalization()(out)
    if need_activate:
        out = keras.layers.ELU()(out)
    return out


def block_1(inputs, filters):
    out = conv_layer(inputs, filters // 4, 1)
    out = conv_layer(out, filters // 4, 3)
    out = conv_layer(out, filters // 4, 3, need_activate=False)
    res = conv_layer(inputs, filters // 4, 1)
    out = keras.layers.ELU()(out + res)
    return out


def block_2(inputs, filters):
    out = conv_layer(inputs, filters // 4, 1)
    out = conv_layer(out, filters // 4, 3)
    out = conv_layer(out, filters // 4, 3, need_activate=False)
    res = conv_layer(inputs, filters // 4, 1)
    out = keras.layers.ELU()(out + res)
    return out


def block_3(inputs, filters):
    out = conv_layer(inputs, filters // 4, 1)
    out = conv_layer(out, filters // 4, 3)
    out = conv_layer(out, filters, 3, need_activate=False)
    res = conv_layer(inputs, filters, 1)
    out = keras.layers.ELU()(out + res)
    return out


def dr_unet(pretrained_weights=None, input_size=(128, 128, 1), dims=32):
    inputs = keras.Input(input_size)
    out = conv_layer(inputs, 16, 1)

    out = block_1(out, dims)
    out_256 = block_3(out, dims)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_256)

    out = block_1(out, dims * 2)
    out_128 = block_3(out, dims * 2)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_128)

    out = block_1(out, dims * 4)
    out_64 = block_3(out, dims * 4)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_64)

    out = block_1(out, dims * 8)
    out_32 = block_3(out, dims * 8)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_32)

    out = block_1(out, dims * 16)
    out_16 = block_3(out, dims * 16)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_16)

    out = block_1(out, dims * 32)
    out = block_3(out, dims * 32)

    up_16 = keras.layers.Conv2DTranspose(filters=dims * 16, kernel_size=2, strides=2, padding='same')(out)
    up = keras.layers.Concatenate()([up_16, out_16])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims * 16)
    up = block_3(up, dims * 16)
    up_32 = keras.layers.Conv2DTranspose(filters=dims * 8, kernel_size=2, strides=2, padding='same')(up)
    up = keras.layers.Concatenate()([up_32, out_32])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims * 8)
    up = block_3(up, dims * 8)
    up_64 = keras.layers.Conv2DTranspose(filters=dims * 4, kernel_size=2, strides=2, padding='same')(up)
    up = keras.layers.Concatenate()([up_64, out_64])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims * 4)
    up = block_3(up, dims * 4)
    up_128 = keras.layers.Conv2DTranspose(filters=dims * 2, kernel_size=2, strides=2, padding='same')(up)
    up = keras.layers.Concatenate()([up_128, out_128])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims * 2)
    up = block_3(up, dims * 2)
    up_256 = keras.layers.Conv2DTranspose(filters=dims * 1, kernel_size=2, strides=2, padding='same')(up)
    up = keras.layers.Concatenate()([up_256, out_256])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims)
    up = block_3(up, dims)
    up = keras.layers.Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='same')(up)
    up = keras.activations.sigmoid(up)

    model = keras.Model(inputs, up)
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=0),
    #               loss=bce_dice_loss,
    #               metrics=['accuracy', dice_loss, jaccard_loss]
    #               )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=0),
                  loss=bce_dice_loss,
                  metrics=['accuracy', jaccard_loss, bce_dice_loss, dice_loss])
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    if __name__ == '__main__':
        # 创建模型
        model = dr_unet()
        model.summary()

        # 生成随机数据作为输入和输出，以测试模型
        import numpy as np

        num_samples = 10
        height, width, channels = 128, 128, 1
        X = np.random.random((num_samples, height, width, channels)).astype(np.float32)
        Y = np.random.randint(0, 2, (num_samples, height, width, channels))


        # 编写一个简单的回调函数来监视损失
        class LossHistory(keras.callbacks.Callback):
            def on_batch_end(self, batch, logs={}):
                if logs.get('loss') is not None:
                    loss = logs.get('loss')
                    if np.isnan(loss) or np.isinf(loss):
                        print("Batch {}: Invalid loss detected.".format(batch))
                    else:
                        print("Batch {}: Loss: {}".format(batch, loss))


        # 初始化回调函数
        history = LossHistory()

        # 训练模型
        model.fit(X, Y, batch_size=2, epochs=1, callbacks=[history])
