import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D
from math import exp, sqrt

class VAE1(tf.keras.Model):
    def __init__(self, input_size, latent_size=15):
        super(VAE1, self).__init__()
        # 1. Encoder
        # 2. Latent Distribution, which includes: Mean Vector & Standard Deviation Vector
        # 3. Sampled Latent Representation
        # 4. Decoder
        self.input_size = input_size  # H*W (28 * 28)
        self.latent_size = latent_size  # Z
        self.hidden_dim = 512  # H_d
        self.encoder = Sequential(
            [
                Flatten(),
                Dense(self.hidden_dim, activation='relu'),
                Dense(self.hidden_dim, activation='relu'),
                Dense(self.hidden_dim, activation='relu'),
            ]
        )

        # Output to self.latent_size because that is what decoder receives
        self.mu_layer = Dense(self.latent_size)
        self.logvar_layer = Dense(self.latent_size)

        self.decoder = Sequential([
            Dense(self.hidden_dim, activation='relu'),
            Dense(self.hidden_dim, activation='relu'),
            Dense(self.hidden_dim, activation='relu'),
            Dense(self.input_size, activation='sigmoid'),
            # make one layer that reshapes to MNIST (1, 28, 28)
            Reshape((1, 28, 28))
            # Reshape((-1, 28, 28))
        ])

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
        x_hat = self.encoder(x)
        mu = self.mu_layer(x_hat)
        logvar = self.logvar_layer(x_hat)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


class CVAE1(tf.keras.Model):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # C
        self.hidden_dim = 4096  # H_d

        self.encoder = Sequential(
            [
                Flatten(),
                Dense(self.hidden_dim, activation='relu', input_dim=self.input_size + self.num_classes),
                Dense(self.hidden_dim, activation='relu'),
                Dense(self.hidden_dim, activation='relu'),
            ]
        )

        # Output to self.latent_size because that is what decoder receives
        self.mu_layer = Dense(self.latent_size, activation='relu')
        self.logvar_layer = Dense(self.latent_size, activation ='relu')

        self.decoder = Sequential([
            Dense(self.hidden_dim, activation='relu', input_dim=self.latent_size + self.num_classes),
            Dense(self.hidden_dim, activation='relu'),
            Dense(self.hidden_dim, activation='relu'),
            Dense(self.input_size, activation='sigmoid'),
            # make one layer that reshapes to MNIST (1, 28, 28)
            Reshape((1, 28, 28))
        ])

    def call(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x = tf.reshape(x, (128, -1))
        x = tf.concat([x, c], axis=1)
        x_hat = self.encoder(x)
        mu = self.mu_layer(x_hat)
        logvar = self.logvar_layer(x_hat)
        z = reparametrize(mu, logvar)
        z = tf.concat([z, c], axis=1)
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    # sample just one?
    epsilon = tf.random.normal(logvar.shape)
    var = tf.exp(logvar)
    sig = tf.sqrt(var)
    z = mu + sig * epsilon
    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    return z


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


def loss_function(x_hat, x, mu, logvar):
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
    rec_loss = bce_function(x_hat, x)
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
    loss = tf.reduce_mean(rec_loss + kl_loss)
    return loss