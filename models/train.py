from preprocessing import load_and_preprocess_images
from vae1 import VAE1
from mapping import Translation
import tensorflow as tf
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_cvae", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
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
    for images,_ in dataset:
        with tf.GradientTape() as tape:
            x_hat, mu, logvar = model(images)
            loss = model.loss_function(x_hat, images, mu, logvar)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss

    return total_loss

def train_vae(model, train_loader, args):
    for epoch_id in range(args.num_epochs):
        total_loss = train_vae_epoch(model, train_loader, args)
        # print(total_loss)
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss / len(train_loader):.6f}")

def train_translation_epoch(translation_model, vae1_model, vae2_model, zipped_data, args):
    total_loss = 0
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)
    for (synethetic, _), (clean, _) in zipped_data:
        with tf.GradientTape() as tape:
            latent_input = vae1_model.encode(synethetic)
            latent_output = vae2_model.encode(clean)
            predicted_output = translation_model(latent_input)
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
    input_height, input_width = 32, 32
    (synthetic_train, synthetic_val), (clean_train, clean_val) = load_and_preprocess_images(folder_path, target_size=(input_height, input_width))

    synthetic_train = tf.data.Dataset.from_tensor_slices((synthetic_train, synthetic_train))
    clean_train = tf.data.Dataset.from_tensor_slices((clean_train, clean_train))

    synthetic_train = synthetic_train.batch(args.batch_size)
    clean_train = clean_train.batch(args.batch_size)

    latent_space = 64
    VAE1_model = VAE1(input_size=input_height * input_width, latent_size=latent_space)
    VAE2_model = VAE1(input_size=input_height * input_width, latent_size=latent_space)
    Translation_model = Translation(input_size=input_height * input_width, latent_size=latent_space)

    #Train VAE1
    train_vae(VAE1_model, synthetic_train, args)

    #Train VAE2
    train_vae(VAE2_model, clean_train, args)

    #Train Translation
    train_translation(Translation_model, VAE1_model, VAE2_model, synthetic_train, clean_train, args)

if __name__ == "__main__":
    args = parseArguments()
    main(args)
