import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from progressbar import ETA, Bar, Percentage, ProgressBar
import time
from scipy.stats import sem

tfd = tfp.distributions

# Load data from CSV
data = pd.read_csv('concatenated_df.csv')
x = data.drop(columns=['treatment', 'y_factual','y_cfactual']).values  # Adjust as necessary to remove non-feature columns
t = data['treatment'].values.reshape(-1, 1)
y = data['y_factual'].values.reshape(-1, 1)

# Configuration variables
reps = 10  # Number of replications
early_stop = 10  # Early stopping criterion
learning_rate = 0.001  # Learning rate for optimizer
epochs = 300  # Number of training epochs
print_every = 10  # Print loss every n epochs
beta = 2.0  # Beta parameter for beta-VAE

# Define model parameters
d = 20  # Latent dimension
h = 200  # Size of hidden layers

# Encoder: takes x as input and outputs the parameters of q(z|x)
class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.hidden = tf.keras.layers.Dense(h, activation='elu')
        self.mu = tf.keras.layers.Dense(latent_dim)
        self.sigma = tf.keras.layers.Dense(latent_dim, activation=tf.nn.softplus)
    
    def call(self, x):
        h = self.hidden(x)
        return self.mu(h), self.sigma(h)

# Decoder: takes z as input and outputs the parameters of p(x|z)
class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        self.hidden = tf.keras.layers.Dense(h, activation='elu')
        self.output_layer = tf.keras.layers.Dense(output_dim)
    
    def call(self, z):
        h = self.hidden(z)
        return self.output_layer(h)

# Initialize encoder and decoder
encoder = Encoder(d)
decoder = Decoder(x.shape[1])

# Define the VAE model
class BetaVAE(tf.keras.Model):
    def __init__(self, latent_dim, beta):
        super(BetaVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(x.shape[1])
        self.beta = beta
    
    def call(self, x):
        # Encoding
        mu, sigma = self.encoder(x)
        qz = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        
        # Sample from q(z|x) for reparameterization
        z = qz.sample()
        
        # Decoding
        x_logit = self.decoder(z)
        
        # Define the distributions for reconstruction and prior
        px_z = tfd.Bernoulli(logits=x_logit)
        pz = tfd.MultivariateNormalDiag(loc=tf.zeros_like(mu), scale_diag=tf.ones_like(sigma))
        
        # Compute losses
        reconstruction_loss = tf.reduce_sum(px_z.log_prob(x), axis=1)
        kl_loss = tf.reduce_sum(tfd.kl_divergence(qz, pz))  # Removed axis=1
        
        # Beta-VAE loss
        total_loss = -tf.reduce_mean(reconstruction_loss - self.beta * kl_loss)
        return total_loss

# Instantiate the model
beta_vae = BetaVAE(d, beta)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Training Loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = beta_vae(x)
    gradients = tape.gradient(loss, beta_vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, beta_vae.trainable_variables))
    
    if epoch % print_every == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Load counterfactual outcomes for PEHE calculation
y_cf = data['y_cfactual'].values.reshape(-1, 1)

# Instantiate the model
beta_vae = BetaVAE(d, beta)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Training Loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = beta_vae(x)
    gradients = tape.gradient(loss, beta_vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, beta_vae.trainable_variables))
    
    if epoch % print_every == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Estimation of treatment and control outcomes
def estimate_outcome(x_input, treatment):
    """Estimate outcome for a given treatment."""
    # Encode the input to get latent distribution parameters
    mu, sigma = beta_vae.encoder(x_input)
    qz = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    
    # Sample from q(z|x)
    z_sample = qz.sample()
    
    # Decode with the sampled z to get predicted y
    y_pred = beta_vae.decoder(z_sample)
    return y_pred

# Predict treatment (t=1) and control (t=0) outcomes for each instance
y_pred_treatment = estimate_outcome(x, treatment=1)
y_pred_control = estimate_outcome(x, treatment=0)

# Calculate PEHE
true_ite = y - y_cf  # True Individual Treatment Effect
pred_ite = y_pred_treatment - y_pred_control  # Predicted Individual Treatment Effect
pehe = np.sqrt(np.mean((pred_ite - true_ite) ** 2))

print(f"PEHE: {pehe:.4f}")
