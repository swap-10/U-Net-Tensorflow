import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image

from train import create_dataset
from unet import UNet

def read_image(image_path: Path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float16)
    return image

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def display(img_list):
  plt.figure(figsize=(10,10))
  title = ['Input Image', 'Predicted Mask']

  for i in range(len(img_list)):
    plt.subplot(1, len(img_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(img_list[i]))
    plt.axis('off')
  plt.show()

def show_predictions(model, image):
    image = image[..., tf.newaxis]
    pred_mask = model.predict(image)
    display([image[0], create_mask(pred_mask)])


def preprocess(image_path="use_default"):
    if image_path == "use_default":
        num = random.randint(0, 2)
        image_path = f"sample_image_{num}.jpg"
    image = read_image(image_path)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Predictions of image segmentation")
    parser.add_argument("--image_path", "-path", type=str, default="use_default", help="Specify image path to run inference on. If unused, one of sample_image_num will be used.")
    parser.add_argument("--out_threshold", "-th", type=float, default=0.5, help="Specify the threshold for prediction.")
    parser.add_argument("--n_classes", "-nc", type=int, default=3, help="Number of classes")
    parser.add_argument("--bilinear", "-b", type=bool, default=False, help="Bilinear of Conv2DTranspose")
    
    return parser.parse_args()


def predict_img(img_path, out_threshold=0.5):
    img = preprocess(img_path)
    model = UNet(n_channels=3, n_classes=args.n_classes, training=False, bilinear=args.bilinear)
    model.build((None, img.shape[0], img.shape[1], img.shape[2]))
    print(model.summary())
    model.load_weights("./saved_model.h5")
    show_predictions(model, img)

if __name__ == "__main__":
    args = get_args()
    predict_img(img_path=args.image_path, out_threshold=args.out_threshold)

