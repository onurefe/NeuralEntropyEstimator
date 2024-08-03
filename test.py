
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import scipy.stats as stats
from neural_entropy_estimator import NeuralEntropyEstimator

# Generate synthetic dataset with hidden linear dependencies
def generate_synthetic_data(num_samples, estimators, dims, latent_dim=10):
    if latent_dim > dims:
        latent_dim = dims
        
    # Step 1: Generate latent variables
    latent_data = np.random.normal(size=(num_samples, latent_dim))
    
    # Step 2: Create a random linear transformation matrix with full rank
    np.random.seed(0)
    transformation_matrix = np.random.randn(estimators, latent_dim, dims)
    
    data = np.einsum("bc, ecd->bed", latent_data, transformation_matrix)

    # Step 4: Add independent noise to ensure full rank covariance
    noise = np.random.normal(scale=0.05, size=(num_samples, estimators, dims))
    data += noise
    
    return data

def build_model(batch_size, estimators, dims):
    init_tensor = np.random.normal(size=(batch_size, estimators, dims))
    model = NeuralEntropyEstimator()
    model(init_tensor)
    return model

@tf.function
def train_step(model, optimizer, X_batch):
    with tf.GradientTape() as tape:
        model(X_batch, training=True)
        loss = tf.reduce_sum(tf.convert_to_tensor(model.layer_losses))
                
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# Train the model
def train_model(model, data, epochs=50, batch_size=32):
    optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='trained_models/epoch_{epoch:02d}.h5',  # Save weights with epoch number in filename
        save_weights_only=True,
        save_freq='epoch'
    )
    
    for epoch in range(epochs):
        for batch in range(0, len(data), batch_size):
            X_batch = data[batch:batch + batch_size]
            
            loss = train_step(model, optimizer, X_batch)
            
        print(model(X_batch)) 
        print(f"Epoch {epoch + 1}, Loss: {loss}")
        # Save weights at the end of the epoch
        checkpoint_callback.model = model
        checkpoint_callback.on_epoch_end(epoch, logs={'loss': loss})

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
    dim = cov_matrix.shape[0]
    return 0.5 * dim * (np.log(2 * np.pi) + 1) + 0.5 * np.log(np.linalg.det(cov_matrix))


# Main script
if __name__ == "__main__":
    # Parameters
    num_samples = 100000
    num_estimators = 2
    num_dims = 16
    
    epochs = 2000
    batch_size = 10000

    # Generate synthetic dataset
    data = generate_synthetic_data(num_samples, num_estimators, num_dims)

    # Build and train the model
    model = build_model(batch_size, num_estimators, num_dims)
    train_model(model, data, epochs=epochs, batch_size=batch_size)
    
    model, predicted_entropy = load_and_test_model("trained_models/epoch_{epoch:02d}.h5".format(epoch=epochs), 
                                                   batch_size=batch_size,
                                                   estimators=num_estimators,
                                                   dims=num_dims,
                                                   test_data=data[0:batch_size, :, :])    

    h_estimation = predicted_entropy
    
    # Calculate true Gaussian entropy for the synthetic data
    cov_matrix1 = np.cov(data[:, 0, :], rowvar=False)
    cov_matrix2 = np.cov(data[:, 1, :], rowvar=False)
    true_entropy = gaussian_entropy(cov_matrix1) + gaussian_entropy(cov_matrix2)
    
    # Evaluate the model
    print(f"True Gaussian Entropy: {true_entropy}")
    print(f"Estimated Entropy: {np.sum(h_estimation)}")