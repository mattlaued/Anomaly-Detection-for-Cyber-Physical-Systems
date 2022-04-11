import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon # returns mean plus std dev x (random) epsilon

class VAE(tf.keras.Model):
    
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # initialise local encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data) # get mean, variance and sample z from encoder
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(data, reconstruction)
            )
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded[-1])
        return decoded

#  Be careful with where vae object points to in the code
def vae_loss(data, reconstruction): #y_true, y_pred
    mu, ln_var, z = vae.encoder(data)
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.mean_squared_error(data, reconstruction)
    )
    kl_loss = 1 + ln_var - tf.square(mu) - tf.exp(ln_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    total_loss = reconstruction_loss + kl_loss
    return total_loss

def sequencify(arr, window):
    '''
    arr: 2D np array where the next row is the data at the next timestep (ie. time dependent data)
    window: sequence length
    '''
    # desired eventual shape
    shape = (arr.shape[0] - window + 1, window, arr.shape[1])
    
    # strides for processor to know how many bytes to skip / stride over to read from memory
    strides = (arr.strides[0],) + arr.strides
    
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def sequencify_y(arr, window):
    
    shape = (arr.shape[0] - window + 1, window,)
    
    # strides for processor to know how many bytes to skip / stride over to read from memory
    strides = (arr.strides[0],) + arr.strides
    
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def plot_data(df):
    for col in df.columns:
        plt.title(col)
        plt.hist(df[col], bins=100)
        plt.show()
