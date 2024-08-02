import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.append("/home/oefe/Code/GroupDenseLayer")

import tensorflow as tf
from neural_entropy_estimator import NeuralEntropyEstimator
import unittest
import numpy as np


class NeuralEntropyEstimatorTest(unittest.TestCase):
    def setUp(self):
        super(NeuralEntropyEstimatorTest, self).setUp()
        x = np.random.normal(size=[500, 1, 3])
        self._layer = NeuralEntropyEstimator(hidden_layer_size=6, num_kernels=1)
        self._layer(x)

    def test_weight_masking_input_to_hidden(self):
        w = np.ones(shape=[1, 3, 6], dtype=np.float32)
        w_masked = self._layer._mask_layer_weights_for_conditional_probability_estimation(w)
        w_masked_expected = np.array([[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0], 
                                       [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

        self.assertTrue(np.allclose(w_masked, w_masked_expected))

    def test_weight_masking_hidden_to_hidden(self):
        w = np.ones(shape=[1, 6, 6], dtype=np.float32)
        w_masked = self._layer._mask_layer_weights_for_conditional_probability_estimation(w)
        w_masked_expected = np.array(
            [[
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]]
        )

        self.assertTrue(np.allclose(w_masked, w_masked_expected))

    def test_weight_masking_hidden_to_output(self):
        w = np.ones(shape=[1, 6, 3], dtype=np.float32)
        w_masked = self._layer._mask_layer_weights_for_conditional_probability_estimation(w)
        w_masked_expected = np.array(
            [[
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ]]
        )

        self.assertTrue(np.allclose(w_masked, w_masked_expected))


unittest.main()
