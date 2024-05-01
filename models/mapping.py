import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D, Dropout, BatchNormalization, LeakyReLU
from math import exp, sqrt

from models.resnet import ResNetBlock

class Translation(tf.keras.Model):
    def __init__(self, input_size, latent_size=512, num_filters=64):
        super(Translation, self).__init__()
        # Will go from one latent space 15 to another latent space 15
        self.input_size = input_size  # H*W (28 * 28)
        self.latent_size = latent_size  # Z
        self.translation = self.make_translation_model(latent_size, num_filters)
        
        
        # Sequential([
        #     Flatten(),
        #     Dense(1024, activation='relu'),
        #     Dense(2048, activation='relu'),
        #     Dense(latent_size, activation='relu'),
        #     Reshape((1, 1, latent_size))
        # ])
    
    def make_translation_model(self, latent_size, dim):
        model = []

         # first 4 convolution layers
        model += [
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"), 
            BatchNormalization(), 
            LeakyReLU(0.3),
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"), 
            BatchNormalization(), 
            LeakyReLU(0.3),
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"), 
            BatchNormalization(), 
            LeakyReLU(0.3),
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"), 
            BatchNormalization(), 
            LeakyReLU(0.3)
        ]

        for i in range(4):
            model += [
                ResNetBlock(input_size=self.input_size, latent_size=latent_size, num_filters=dim)
            ]
        
        model += [
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"), 
            BatchNormalization(), 
            LeakyReLU(0.3),
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"), 
            BatchNormalization(), 
            LeakyReLU(0.3),
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"), 
            BatchNormalization(), 
            LeakyReLU(0.3),
        ]

        model += [
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"),
            Flatten(),
            Dense(latent_size),
            Reshape((1, 1, latent_size))
        ]

    def call(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        
        return self.translation(x)
    
    def loss_function(self, latent_result, latent_target):
        """
        Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
        Returned loss is the average loss per sample in the current batch.

        Inputs:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - x: Input data for this timestep of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
        
        Returns:
        - loss: Tensor containing the scalar loss for the negative variational lowerbound
        """
        # find each loss
        # what is the loss function here?
        return tf.reduce_mean(tf.keras.losses.MSE(latent_target, latent_result))


def bce_function(x_hat, x):
    """
    Computes the reconstruction loss of the VAE.
    
    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    
    Returns:
    - reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
    """
    bce_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.SUM,
    )
    reconstruction_loss = bce_fn(x, x_hat) * x.shape[
        -1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
    return reconstruction_loss
