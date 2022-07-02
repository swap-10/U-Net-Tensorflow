import tensorflow as tf
from .unet_utils import *

class UNet(tf.keras.Model):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inp = DoubleConvModule(64)
        
        self.down1 = Downscale(128)
        self.down2 = Downscale(256)
        self.down3 = Downscale(512)
        self.down4 = Downscale(1024)
        
        self.up1 = Upscale(512, bilinear=bilinear)
        self.up2 = Upscale(256, bilinear=bilinear)
        self.up3 = Upscale(128, bilinear=bilinear)
        self.up4 = Upscale(64, bilinear=bilinear)

        self.out = ClassifyConv(n_classes)

    def call(self, input_tensor):
        x1 = self.inp(input_tensor)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xf = self.down4(x4)
        
        xf = self.up1(xf, x4)
        xf = self.up2(xf, x3)
        xf = self.up3(xf, x2)
        xf = self.up4(xf, x1)
        
        x = self.out(xf)

        return x

    def build_graph(self):
        x = tf.keras.Input(shape=(572, 572, 3))
        return tf.keras.Model(inputs=x, outputs=self.call(x))