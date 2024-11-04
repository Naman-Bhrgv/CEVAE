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

# Define model parameters
d1, d2 = 20, 10  # Dimensions of latent variables z1 and z2
h = 200  # Size of hidden layers

# Encoder for z1, conditioned on z2
class EncoderZ1(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(EncoderZ1, self).__init__()
        self.hidden = tf.keras.layers.Dense(h, activation='elu')
        self.mu = tf.keras.layers.Dense(latent_dim)
        self.sigma = tf.keras.layers.Dense(latent_dim, activation=tf.nn.softplus)
    
    def call(self, x, z2):
        h = self.hidden(tf.concat([x, z2], axis=1))
        return self.mu(h), self.sigma(h)

# Encoder for z2
class EncoderZ2(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(EncoderZ2, self).__init__()
        self.hidden = tf.keras.layers.Dense(h, activation='elu')
        self.mu = tf.keras.layers.Dense(latent_dim)
        self.sigma = tf.keras.layers.Dense(latent_dim, activation=tf.nn.softplus)
    
    def call(self, x):
        h = self.hidden(x)
        return self.mu(h), self.sigma(h)

# Decoder: conditioned on both z1 and z2
class HierarchicalDecoder(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(HierarchicalDecoder, self).__init__()
        self.hidden = tf.keras.layers.Dense(h, activation='elu')
        self.output_layer = tf.keras.layers.Dense(output_dim)
    
    def call(self, z1, z2):
        h = self.hidden(tf.concat([z1, z2], axis=1))
        return self.output_layer(h)

# Define the Hierarchical VAE model
class HierarchicalVAE(tf.keras.Model):
    def __init__(self, latent_dim1, latent_dim2):
        super(HierarchicalVAE, self).__init__()
        self.encoder_z2 = EncoderZ2(latent_dim2)
        self.encoder_z1 = EncoderZ1(latent_dim1)
        self.decoder = HierarchicalDecoder(x.shape[1])
    
    def call(self, x):
        # Encoding for z2
        mu_z2, sigma_z2 = self.encoder_z2(x)
        qz2 = tfd.MultivariateNormalDiag(loc=mu_z2, scale_diag=sigma_z2)
        z2 = qz2.sample()
        
        # Encoding for z1, conditioned on z2
        mu_z1, sigma_z1 = self.encoder_z1(x, z2)
        qz1 = tfd.MultivariateNormalDiag(loc=mu_z1, scale_diag=sigma_z1)
        z1 = qz1.sample()
        
        # Decoding
        x_logit = self.decoder(z1, z2)
        
        # Define the distributions for reconstruction and priors
        px_z = tfd.Bernoulli(logits=x_logit)
        pz2 = tfd.MultivariateNormalDiag(loc=tf.zeros_like(mu_z2), scale_diag=tf.ones_like(sigma_z2))
        pz1 = tfd.MultivariateNormalDiag(loc=tf.zeros_like(mu_z1), scale_diag=tf.ones_like(sigma_z1))
        
        # Compute losses
        reconstruction_loss = tf.reduce_sum(px_z.log_prob(x), axis=1)
        kl_loss_z2 = tf.reduce_sum(tfd.kl_divergence(qz2, pz2))
        kl_loss_z1 = tf.reduce_sum(tfd.kl_divergence(qz1, pz1))
        
        # Total loss with KL losses for both layers
        total_loss = -tf.reduce_mean(reconstruction_loss - (kl_loss_z2 + kl_loss_z1))
        return total_loss

# Instantiate the model
hierarchical_vae = HierarchicalVAE(d1, d2)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Training Loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = hierarchical_vae(x)
    gradients = tape.gradient(loss, hierarchical_vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, hierarchical_vae.trainable_variables))
    
    if epoch % print_every == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Define the function to estimate treatment and control outcomes
def estimate_outcome(x_input, treatment):
    mu_z2, sigma_z2 = hierarchical_vae.encoder_z2(x_input)
    qz2 = tfd.MultivariateNormalDiag(loc=mu_z2, scale_diag=sigma_z2)
    z2_sample = qz2.sample()
    
    mu_z1, sigma_z1 = hierarchical_vae.encoder_z1(x_input, z2_sample)
    qz1 = tfd.MultivariateNormalDiag(loc=mu_z1, scale_diag=sigma_z1)
    z1_sample = qz1.sample()
    
    y_pred = hierarchical_vae.decoder(z1_sample, z2_sample)
    return y_pred

# Predict treatment and control outcomes
y_pred_treatment = estimate_outcome(x, treatment=1)
y_pred_control = estimate_outcome(x, treatment=0)

# Load counterfactual outcomes for PEHE calculation
y_cf = data['y_cfactual'].values.reshape(-1, 1)
# Calculate PEHE
true_ite = y - y_cf
pred_ite = y_pred_treatment - y_pred_control
pehe = np.sqrt(np.mean((pred_ite - true_ite) ** 2))

print(f"PEHE: {pehe:.4f}")

# Calculate ATE
ate = tf.reduce_mean(y_pred_treatment - y_pred_control).numpy()
print(f"ATE: {ate:.4f}")
