
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import scipy.stats as stats
from neural_probability_estimator import NeuralProbabilityEstimator
from probability_estimator import ProbabilityEstimator, ProbabilityEstimatorCovarianceDiagonalizingKernel
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
    latent_data = 5. * np.random.normal(size=(num_samples, latent_dim))
    
    # Step 2: Create a random linear transformation matrix with full rank
    np.random.seed(seed)
    transformation_matrix = np.random.randn(estimators, latent_dim, dims)
    
    data = np.einsum("bc, ecd->bed", latent_data, transformation_matrix)

    # Step 4: Add independent noise to ensure full rank covariance
    noise = np.random.normal(scale=1.0, size=(num_samples, estimators, dims))
    data += noise
    
    return data

def build_neural_model(batch_size, estimators, dims):
    init_tensor = np.random.normal(size=(batch_size, estimators, dims))
    model = NeuralProbabilityEstimator()
    model(init_tensor)
    return model

def build_kernel_model(batch_size, estimators, dims):
    init_tensor = np.random.normal(size=(batch_size, estimators, dims))
    model = ProbabilityEstimator()
    model(init_tensor)
    return model

def build_kernel_model_diagonalizing_kernel(batch_size, estimators, dims):
    init_tensor = np.random.normal(size=(batch_size, estimators, dims))
    model = ProbabilityEstimatorCovarianceDiagonalizingKernel()
    model(init_tensor)
    return model

@tf.function
def train_step(model, optimizer, X_batch):
    with tf.GradientTape() as tape:
        p = model(X_batch, training=True)
        loss = tf.reduce_sum(tf.convert_to_tensor(model.layer_losses))
                
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        h = -tf.reduce_mean(tf.math.log(p), axis=0)
        
    return loss, h

# Train the model
def train_model(model, data, epochs=50, batch_size=32):
    data1, data2 = data
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-3)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='trained_models/epoch_{epoch:02d}.h5',  # Save weights with epoch number in filename
        save_weights_only=True,
        save_freq='epoch'
    )
    
    true_entropies = []
    estimated_entropies = []
    
    for epoch in range(epochs):
        if epoch < epochs // 2:
            data = data1
        else:
            data = data2
            
        for batch in range(0, len(data), batch_size):
            X_batch = data[batch:batch + batch_size]

            cov_matrix = np.cov(X_batch[:, 0, :], rowvar=False)
            loss, estimated_entropy = train_step(model, optimizer, X_batch)
            true_entropy = gaussian_entropy(cov_matrix)
        
            true_entropies.append(true_entropy)
            estimated_entropies.append(estimated_entropy[0])
            
        # Evaluate the model
        print(f"True Gaussian Entropy: {true_entropy}")
        print(f"Estimated Entropy: {estimated_entropy}")
        print(f"Epoch {epoch + 1}, Loss: {loss}")
        
        # Save weights at the end of the epoch
        checkpoint_callback.model = model
        checkpoint_callback.on_epoch_end(epoch, logs={'loss': loss})

    # Convert entropies to array.
    true_entropies = np.array(true_entropies)
    estimated_entropies = np.array(estimated_entropies)
        
    return true_entropies, estimated_entropies

# Load weights and test the model
def load_and_test_model(weights_filepath, batch_size, estimators, dims, test_data):
    # Initialize model
    model = build_model(batch_size, estimators, dims)
    
    # Load weights
    model.load_weights(weights_filepath)
    
    # Predict entropy on test data
    predicted_entropy = model(test_data)
    return model, predicted_entropy

# Gaussian entropy approximation
def gaussian_entropy(cov_matrix):
    if len(np.shape(cov_matrix)) == 0:
        return 0.5 * (np.log(2 * np.pi) + 1) + 0.5 * np.log(cov_matrix)
    else:
        dim = cov_matrix.shape[0]
        return 0.5 * dim * (np.log(2 * np.pi) + 1) + 0.5 * np.log(np.linalg.det(cov_matrix))


# Main script
if __name__ == "__main__":
    # Parameters
    num_samples = 100000
    num_estimators = 1
    num_dims = 2
    
    epochs = 500
    batch_size = 10000

    # Generate synthetic dataset
    data1 = generate_synthetic_data(num_samples, num_estimators, num_dims, seed=0, latent_dim=num_dims-1)
    data2 = generate_synthetic_data(num_samples, num_estimators, num_dims, seed=1, latent_dim=num_dims-1)
    
    # Build and train the model
    # model = build_neural_model(batch_size, num_estimators, num_dims)
    # model = build_kernel_model(batch_size, num_estimators, num_dims)
    model = build_kernel_model_diagonalizing_kernel(batch_size, num_estimators, num_dims)
    true_entropies, estimated_entropies = train_model(model, [data1, data2], epochs=epochs, batch_size=batch_size)
    
    data = pd.DataFrame({'True entropies': true_entropies, 'Estimated entropies': estimated_entropies})

    # Save the DataFrame to a CSV file
    data.to_csv('transient_entropies_diagonal_kernel.csv', index=False)