import tensorflow as tf
from unet import UNet
from pathlib import Path
import argparse

def read_image(image_path: Path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3, dtype=tf.float32)
    return image

def preprocess(image_path, mask_path):
    image = read_image(image_path)
    mask = read_image(mask_path)
    image = tf.image.resize(image, [300, 300])
    mask = tf.image.resize(mask, [300, 300])
    return image, mask

def create_dataset(img_dir, masks_dir, mask_suffix="", batch_size=16):
    image_paths = list(Path.glob(img_dir, "[!.]*"))
    mask_paths = list(Path.glob(masks_dir, "[!.]*"+mask_suffix + "*"))
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(preprocess)

    return dataset

def get_args():
    parser = argparse.ArgumentParser(description="Train the U-Net")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", "-lr", dest="lr", type=float, default=1e-5, help="Learning Rate")
    parser.add_argument("--n_classes", "-nc", type=int, default=2, help="Number of classes")
    parser.add_argument("--mask_suffix", "-ms", type=str, default="", help="Mask image names of the form: 'image_name_mask.ext")
    parser.add_argument("--bilinear", "-bi", type=bool, default=False, help="True: use bilinear upsampling; False: Use Conv2DTranspose. Default: False")
        

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('accuracy') >= 0.93:
      print("Reached 93% accuracy so stopping training\n")
      self.model.stop_training=True


if __name__ == "__main__":
    args = get_args()
    train_images_dir = Path("./data/train/images/")
    train_masks_dir = Path("./data/train/masks/")

    test_images_dir = Path("./data/train/images/")
    test_masks_dir = Path("./data/train/masks/")

    train_dataset = create_dataset(
        train_images_dir,
        train_masks_dir,
        mask_suffix=args.mask_suffix,
        batch_size=args.batch_size
        )

    test_dataset = create_dataset(
        test_images_dir,
        test_masks_dir,
        mask_suffix=args.mask_suffix,
        batch_size=args.batch_size
        )


    model = UNet(n_channels=3, n_classes=args.n_classes, bilinear=args.bilinear)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
    early_stopping_callback = EarlyStoppingCallback()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./",
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch'
    )

    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=test_dataset,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        verbose=1
    )
