import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D, Dropout, BatchNormalization, ReLU, LeakyReLU
from math import exp, sqrt

class ResNetBlock(tf.keras.Model):
    def __init__(self, input_size, latent_size=512, num_filters=64):
        super(ResNetBlock, self).__init__()
        # Will go from one latent space 15 to another latent space 15
        self.input_size = input_size  # H*W (28 * 28)
        self.latent_size = latent_size  # Z

        self.conv_block = self.build_conv_block(input_size=input_size, latent_size=latent_size, dim=num_filters)

    def build_conv_block(self, input_size, latent_size, dim):
        conv_block = []

        # "ResNet Blocks"
        conv_block += [
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"), 
            BatchNormalization(), 
            LeakyReLU(0.3),
            Conv2D(filters=dim, kernel_size=3, strides=1, padding="same"),
            BatchNormalization(),
        ]

        return Sequential(conv_block)



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
        print(x.shape)
        y = self.conv_block(x)
        print(y.shape)
        return x + y
    
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
        print("target", latent_target)
        print("result", latent_result)
        return tf.reduce_mean(tf.keras.losses.MAE(latent_target, latent_result))
        # return tf.reduce_mean(tf.keras.losses.MSE(latent_target, latent_result))

