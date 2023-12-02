import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from patchify import patchify
import tensorflow as tf
from train import load_data, tf_dataset
from vit import ViT
from hyperparameters import hp

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Paths """
    dataset_path = "input"
    model_path = os.path.join("files", "model.h5")

    """ Dataset """
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    test_ds = tf_dataset(test_x, batch=hp["batch_size"])

    """ Model """
    model = ViT(hp)
    model.load_weights(model_path)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        metrics=["acc"]
    )

    model.evaluate(test_ds)
