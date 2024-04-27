import tensorflow as tf
import keras

# netG in Pix2Pix architecture. Generator network.
class netG(tf.keras.Model):
    def __init__(self, input_nc, output_nc, ngf=64, k_size=3, n_downsampling=8, norm_layer=keras.layers.BatchNormalization(axis = -1), padding_type="reflect", opt=None, **kwargs):
        
        super(netG, self).__init__()
        activation = keras.layers.ReLU()
        model = keras.Sequential([
            keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')),
            keras.layers.Conv2D(filters = min(ngf, opt.mc), kernel_size = 7, padding = 'valid')
            norm_layer,
            activation,
        ])


        
