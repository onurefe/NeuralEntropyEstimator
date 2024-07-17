import unittest
import tensorflow as tf
from math import pi
from tensorflow.keras.initializers import GlorotUniform, Zeros
from neural_entropy_estimator import NeuralEntropyEstimator

class TestNeuralEntropyEstimator(unittest.TestCase):

    def test_initialization(self):
        model = NeuralEntropyEstimator()
        self.assertIsNotNone(model, "Model initialization failed.")

    def test_shape_consistency(self):
        model = NeuralEntropyEstimator()
        input_shape = (32, 33)  # Example input shape (batch_size, n_dims)
        model.build(input_shape)
        self.assertEqual(model._input_shape, input_shape, "Input shape mismatch.")
        self.assertEqual(model._n_dims, 33, "Dimension mismatch.")
        self.assertEqual(model._batch_size, 32, "Batch size mismatch.")

    def test_forward_pass(self):
        model = NeuralEntropyEstimator()
        input_shape = (32, 33)
        model.build(input_shape)
        x = tf.random.normal(input_shape)
        h_merged, h_estimation, h_upper_bound = model(x, training=False)
        self.assertEqual(h_merged.shape, (33), "Output shape mismatch for h_merged.")
        self.assertEqual(h_estimation.shape, (33), "Output shape mismatch for h_estimation.")
        self.assertEqual(h_upper_bound.shape, (33), "Output shape mismatch for h_upper_bound.")

    def test_loss_calculation(self):
        model = NeuralEntropyEstimator()
        input_shape = (32, 33)
        model.build(input_shape)
        x = tf.random.normal(input_shape)
        _, h_estimation, _ = model(x, training=True)
        
        entropy_reg = model._compute_entropy_regularization(h_estimation)
        covariance_reg = model._compute_input_map_covariance_diagonalization_regularization(x)
        unitarity_reg = model._compute_input_map_unitarity_regularization()
        l2_reg = model._compute_l2_regularization()
        
        self.assertGreaterEqual(entropy_reg.numpy(), 0, "Entropy regularization loss should be non-negative.")
        self.assertGreaterEqual(covariance_reg.numpy(), 0, "Covariance diagonalization loss should be non-negative.")
        self.assertGreaterEqual(unitarity_reg.numpy(), 0, "Unitarity regularization loss should be non-negative.")
        self.assertGreaterEqual(l2_reg.numpy(), 0, "L2 regularization loss should be non-negative.")

    def test_sorting_function(self):
        model = NeuralEntropyEstimator()
        input_shape = (32, 33)
        model.build(input_shape)
        x = tf.random.normal(input_shape)
        
        model(x, training=True)
        
        # Now applying the model second time, entropies should be sorted out.
        _, _, h_upper_bound = model(x, training=True)
        
        for i in range(len(h_upper_bound) - 1):
            self.assertGreaterEqual(h_upper_bound[i], h_upper_bound[i + 1], "Entropies should be sorted in descending order.")

if __name__ == "__main__":
    unittest.main()
