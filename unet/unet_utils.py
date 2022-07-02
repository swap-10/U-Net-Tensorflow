import tensorflow as tf


class DoubleConvModule(tf.keras.layers.Layer):


    def __init__(self, filters, mid_channels=None):
        super(DoubleConvModule, self).__init__()
        if not mid_channels:
            mid_channels = filters
        self.conv1 = tf.keras.layers.Conv2D(
            filters=mid_channels,
            kernel_size=(3,3),
            padding='valid',
            )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3,3),
            padding='valid'
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        return x


class Downscale(tf.keras.layers.Layer):


    def __init__(self, filters):
        super(Downscale, self).__init__()
        self.mp2d_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.doubleconv = DoubleConvModule(filters)
    
    def call(self, input_tensor, training=False):
        x = self.mp2d_1(input_tensor)
        return self.doubleconv(x, training=training)


class Upscale(tf.keras.layers.Layer):


    def __init__(self, filters, bilinear=False):
        super().__init__()
        self.filters = filters
        if bilinear:
            self.upsample = tf.keras.layers.UpSampling2D(
                size=(2,2),
                interpolation='bilinear')
        else:
            self.upsample = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2), padding='valid')
        self.doubleconv = DoubleConvModule(filters)

    def call(self, input_tensor, skip_connection):
        x1 = self.upsample(input_tensor)
        x2 = skip_connection
        x2 = tf.image.random_crop(x2, size=(x2.shape[0], x1.shape[1], x1.shape[2], self.filters))
        x = tf.keras.layers.Concatenate()([x1, x2])
        return self.doubleconv(x)


class ClassifyConv(tf.keras.layers.Layer):

    
    def __init__(self, filters):
        super(ClassifyConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=(1,1))
    
    def call(self, input_tensor):
        return self.conv(input_tensor)