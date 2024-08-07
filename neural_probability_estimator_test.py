import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.append("/home/oefe/Code/GroupDenseLayer")

import tensorflow as tf
from neural_probability_estimator import NeuralProbabilityEstimator
import unittest
import numpy as np


class NeuralProbabilityEstimatorTest(unittest.TestCase):
    def setUp(self):
        super(NeuralProbabilityEstimatorTest, self).setUp()
        x = np.random.normal(size=[500, 2, 3])
        self._layer = NeuralProbabilityEstimator(hidden_layer_size=6, num_kernels=1)
        self._layer(x)

    def test_weight_masking_input_to_hidden(self):
        w = np.ones(shape=[6, 12], dtype=np.float32)
        w_masked = self._layer._mask_layer_weights_for_conditional_probability_estimation(w)
        w_masked_expected = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                      [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], 
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertTrue(np.allclose(w_masked, w_masked_expected))

    def test_weight_masking_hidden_to_hidden(self):
        w = np.ones(shape=[6, 6], dtype=np.float32)
        w_masked = self._layer._mask_layer_weights_for_conditional_probability_estimation(w)
        w_masked_expected = np.array(
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 1.0, 0.0], 
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        self.assertTrue(np.allclose(w_masked, w_masked_expected))

    def test_weight_masking_hidden_to_output(self):
        w = np.ones(shape=[12, 6], dtype=np.float32)
        w_masked = self._layer._mask_layer_weights_for_conditional_probability_estimation(w)
        w_masked_expected = np.array(
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        self.assertTrue(np.allclose(w_masked, w_masked_expected))


unittest.main()
