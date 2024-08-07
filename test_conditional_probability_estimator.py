
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import scipy.stats as stats
from neural_conditional_probability_estimator import NeuralConditionalProbabilityEstimator
from neural_conditional_probability_estimator_multiple import NeuralConditionalProbabilityEstimatorMultiple
import pandas as pd
import matplotlib.pyplot as plt

def plot(data):
    plt.figure(figsize=(8, 6))
    plt.plot(data.index, data['True entropies'], label='True entropy', color='blue', marker='o')
    plt.plot(data.index, data['Estimated entropies'], label='Estimated entropy', color='red', marker='x')

    # Adding labels and title
    plt.xlabel('Step')
    plt.ylabel('Entropy')
    plt.title('True')
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.savefig("entropies.png")
    
# Generate synthetic dataset with hidden linear dependencies
def generate_synthetic_data(num_samples, estimators, dims, latent_dim=10, seed=0):
    if latent_dim > dims:
        latent_dim = dims
        
    # Step 1: Generate latent variables
    np.random.seed(seed)
    latent_data = 10. * np.random.normal(size=(num_samples, latent_dim))
    
    # Step 2: Create a random linear transformation matrix with full rank
    np.random.seed(seed+100)
    transformation_matrix = np.random.randn(estimators, latent_dim, dims)
    
    data = np.einsum("bc, ecd->bed", latent_data, transformation_matrix)

    # Step 4: Add independent noise to ensure full rank covariance
    np.random.seed(seed+200)
    noise = np.random.normal(scale=1., size=(num_samples, estimators, dims))
    data += noise
    
    return data

def build_model(batch_size, estimators, dims):
    init_tensor = np.random.normal(size=(batch_size, estimators, dims))
    model = NeuralConditionalProbabilityEstimator()
    model([init_tensor, init_tensor])
    return model

def build_model_multiple(batch_size, estimators, dims):
    init_tensor = np.random.normal(size=(batch_size, estimators, dims))
    model = NeuralConditionalProbabilityEstimatorMultiple()
    model([init_tensor, init_tensor])
    return model

@tf.function
def train_step(model, optimizer, X_batch):
    with tf.GradientTape() as tape:
        p_cond = model([X_batch, X_batch], training=True)
        loss = tf.reduce_sum(tf.convert_to_tensor(model.layer_losses))
                
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        h_cond = -tf.reduce_mean(tf.math.log(p_cond), axis=0)
        
    return loss, h_cond

# Train the model
def train_model(model, data, epochs=50, batch_size=32):
    data1, data2 = data
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.5e-3)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='trained_models/epoch_{epoch:02d}.h5',  # Save weights with epoch number in filename
        save_weights_only=True,
        save_freq='epoch'
    )
    
    estimated_entropies = []

    for epoch in range(epochs):
        if epoch < epochs // 2:
            data = data1
        else:
            data = data2
            
        for batch in range(0, len(data), batch_size):
            X_batch = data[batch:batch + batch_size]
            loss, h_cond_estimated = train_step(model, optimizer, X_batch)
            
            mask = 1. - np.eye(33, dtype=np.float32)
            estimated_entropies.append(h_cond_estimated * mask)
            
        # Evaluate the model
        print(f"Mean estimated entropy: {np.mean(h_cond_estimated * mask)}")
        print(f"Epoch {epoch + 1}, Loss: {loss}")
        
        # Save weights at the end of the epoch
        checkpoint_callback.model = model
        checkpoint_callback.on_epoch_end(epoch, logs={'loss': loss})

    # Convert entropies to array.
    estimated_entropies = np.array(estimated_entropies)
        
    return estimated_entropies

# Gaussian entropy approximation
def gaussian_entropy(cov_matrix):
    dim = cov_matrix.shape[-1]
    
    # Retrieve the low rank elements.
    low_rank_cov = np.einsum("nncd->ncd", cov_matrix)
    
    h_joint = 0.5 * dim * (np.log(2 * np.pi) + 1) + 0.5 * np.log(np.linalg.det(cov_matrix))
    h_joint = np.nan_to_num(h_joint, nan=0.0, posinf=0.0, neginf=0.0)
    
    h_marginal = 0.5 * (np.log(2 * np.pi) + 1) + 0.5 * np.log(np.abs(low_rank_cov[:, 0, 0]))
    
    h_joint = h_joint + np.diag(h_marginal)
    
    return h_joint

# Main script
if __name__ == "__main__":
    # Parameters
    num_samples = 660000 
    num_estimators = 33
    num_dims = 1
    
    epochs = 25
    batch_size = 16500

    # Generate synthetic dataset
    data1 = generate_synthetic_data(num_samples, num_estimators, num_dims, latent_dim=25, seed=0)
    data2 = generate_synthetic_data(num_samples, num_estimators, num_dims, latent_dim=25, seed=1)
    
    # Build and train the model
    # model = build_model(batch_size, num_estimators, num_dims)
    model_multiple = build_model_multiple(batch_size, num_estimators, num_dims)
    
    # estimated_entropies = train_model(model, [data1, data2], epochs=epochs, batch_size=batch_size)
    estimated_entropies_multiple = train_model(model_multiple, [data1, data2], epochs=epochs, batch_size=batch_size)
    
    """
    df1 = pd.DataFrame({})
    
    for l in range(10):
        given = np.random.random_integers(0, 32)
        estimated = np.random.random_integers(0, 32)
        
        df1["given{}_estimated{}".format(given, estimated)] = estimated_entropies[:, given, estimated]    
        
    df1.to_csv("neural_conditional_pdf.csv")
    """

    df2 = pd.DataFrame({})
    
    for l in range(10):
        given = np.random.random_integers(0, 32)
        estimated = np.random.random_integers(0, 32)
        
        df2["given{}_estimated{}".format(given, estimated)] = estimated_entropies_multiple[:, given, estimated]    
        
    df2.to_csv("neural_conditional_pdf_multiple.csv")