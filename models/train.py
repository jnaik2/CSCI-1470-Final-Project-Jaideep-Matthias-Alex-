from resnet import ResNetBlock
from preprocessing import load_and_preprocess_images
import argparse
from vae1 import VAE1
from mapping import Translation
import tensorflow as tf
from pix2pix import Pix2PixModel
import os
import cv2

IMAGE_DIM = 256


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_cvae", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--latent_size", type=int, default=1024)
    parser.add_argument("--input_size", type=int, default= 256 * 256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    return args

def train_vae_epoch(model, dataset, args):
    """
    Train your VAE with one epoch.

    Inputs:
    - model: Your VAE instance.
    - train_loader: A tf.data.Dataset of MNIST dataset.
    - args: All arguments.
    - is_cvae: A boolean flag for Conditional-VAE. If your model is a Conditional-VAE,
    set is_cvae=True. If it's a Vanilla-VAE, set is_cvae=False.

    Returns:
    - total_loss: Sum of loss values of all batches.
    """    
    total_loss = 0
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)
    for images in dataset:
        with tf.GradientTape() as tape:
            x_hat, mu, logvar = model(images)
            loss = model.loss_function(x_hat, images, mu, logvar)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += (loss/args.batch_size)

    return total_loss

def train_vae(model, train_loader, args):
    for epoch_id in range(args.num_epochs):
        total_loss = train_vae(model, train_loader, args, is_cvae=args.is_cvae)
        print(total_loss)
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(train_loader):.6f}")

def train_translation_epoch(translation_model, vae1_model, vae2_model, zipped_data, args):
    total_loss = 0
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)
    for synethetic, clean in zipped_data:
        with tf.GradientTape() as tape:
            latent_input = vae1_model.encode(synethetic)
            latent_output = vae2_model.encode(clean)
            predicted_output = translation_model(latent_input)
            # print(latent_output.shape)
            # print(predicted_output.shape)
            loss = translation_model.loss_function(predicted_output, latent_output)
        gradients = tape.gradient(loss, translation_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, translation_model.trainable_variables))
        total_loss += (loss/args.batch_size)
    return total_loss

def train_translation(translation_model, vae1_model, vae2_model, synthetic_train, clean_train, args):
    zipped_data = zip(synthetic_train, clean_train)
    for epoch_id in range(args.num_epochs):
        total_loss = train_translation_epoch(translation_model, vae1_model, vae2_model, zipped_data, args)
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(synthetic_train):.6f}")


def main(args):
    folder_path = "../data/Flickr2K"
    (synthetic_train, synthetic_val), (clean_train, clean_val) = load_and_preprocess_images(folder_path, target_size=(IMAGE_DIM, IMAGE_DIM))
    use_pix = True

    synthetic_train = tf.data.Dataset.from_tensor_slices((synthetic_train, synthetic_train))
    clean_train = tf.data.Dataset.from_tensor_slices((clean_train, clean_train))

    synthetic_train = synthetic_train.batch(args.batch_size)
    clean_train = clean_train.batch(args.batch_size)
    
    if use_pix:
        pix2pix_model_1 = Pix2PixModel()
        Translation_model = Translation(input_size=IMAGE_DIM**2, latent_size=512)
        pix2pix_model_2 = Pix2PixModel()

        pix2pix_model_1.fit(synthetic_train, clean_train, 25)
        pix2pix_model_2.fit(clean_train, synthetic_train, 25)

        train_translation(Translation_model, pix2pix_model_1, pix2pix_model_2, synthetic_train, clean_train, args)

        res = pix2pix_model_1.predict(synthetic_val)
        res2 = pix2pix_model_2.predict(clean_val)
        res3 = pix2pix_model_2.decode(Translation_model(pix2pix_model_1.encode(synthetic_val)))
        for i in range(10):
            cv2.imshow('syn', synthetic_val[i])
            cv2.imshow('rec syn', res[i].numpy())
            # print("VAL", synthetic_val[i])
            # print("REC", res[i])
            cv2.imshow('clean', clean_val[i])
            cv2.imshow('rec clean', res2[i].numpy())
            cv2.imshow('REC', res3[i].numpy())
            cv2.waitKey(0)

    else:
        VAE1_model = VAE1(input_size=IMAGE_DIM**2, latent_size=1024)
        Translation_model = Translation(input_size=IMAGE_DIM**2, latent_size=1024)
        VAE2_model = VAE1(input_size=IMAGE_DIM**2, latent_size=1024)
        train_vae(VAE1_model, synthetic_train, args)

        #Train VAE2
        train_vae(VAE2_model, clean_train, args)

        #Train Translation
        train_translation(Translation_model, VAE1_model, VAE2_model, synthetic_train, clean_train, args)

if __name__ == "__main__":
    args = parseArguments()
    main(args)

    



