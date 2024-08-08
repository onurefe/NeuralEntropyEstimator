import tensorflow
import tensorflow as tf
from tensorflow.keras.initializers import Constant, RandomUniform
from math import pi

class ProbabilityEstimator(tensorflow.keras.Model):
    def __init__(
        self,
        num_kernels=4,
        weight_logits_softmax_gain=7.5,
        max_log_variance_magnitude=7.5,
        eps=1e-7,
        name="probability_estimator",
        *args,
        **kwargs
    ):
        self._num_kernels = num_kernels
        self._weight_logits_softmax_gain = weight_logits_softmax_gain
        self._max_log_variance_magnitude = max_log_variance_magnitude
        self._eps = eps

        super(ProbabilityEstimator, self).__init__(name=name)

    def get_config(self):
        config = super(ProbabilityEstimator, self).get_config()
        config.update({"num_kernels": self._num_kernels, "eps": self._eps})
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape

        self.layer_losses = []
        self._create_kernel_centers()
        self._create_weight_logits()
        self._create_log_variance()

    def call(self, x, training=False):
        p = self._estimate_probabilities(x)

        if training:
            h = self._compute_entropy(p)
            self._add_regularizations(h)

        return p

    def _add_regularizations(self, h):
        h_mean = tf.reduce_mean(h)
        
        self.layer_losses.clear()
        self.layer_losses.append(h_mean)

    def _compute_entropy(self, p):
        p = tf.maximum(p, self._eps)
        h = -tf.reduce_mean(tf.math.log(p), axis=0)

        return h

    def _estimate_probabilities(self, x):
        x = self._subtract_kernel_centers(x)
        gaussian_probability_estimations = self._gaussian1d(x)

        p = self._mix_gaussian_probability_estimations(gaussian_probability_estimations)
        return p

    def _mix_gaussian_probability_estimations(self, gaussian_probability_estimations):
        # Since each kernel has different transformation, probabilities
        # for two axis should be converted to joint probability
        # before merging kernels.
        weights = self._compute_gaussian_kernel_weights()

        lgpe = tf.math.log(gaussian_probability_estimations + self._eps)
        lgpe = tf.reduce_sum(lgpe, axis=-1)
        p_kernels = tf.exp(lgpe)
        p = tf.reduce_sum(p_kernels * weights[tf.newaxis, ...], axis=-1)

        return p

    def _compute_gaussian_kernel_weights(self):
        clipped_weight_logits = tf.clip_by_value(self._weight_logits, 
                                                 -self._weight_logits_softmax_gain, 
                                                 self._weight_logits_softmax_gain)
        
        return tf.nn.softmax(clipped_weight_logits, axis=-1)        

    def _compute_variance(self):
        clipped_log_var = tf.clip_by_value(self._log_variance, -self._max_log_variance_magnitude, self._max_log_variance_magnitude)
        var = tf.exp(clipped_log_var)
        return var

    def _subtract_kernel_centers(self, x):
        # Add kernel dimension to x.
        x = x[:, :, tf.newaxis, :]

        # Add batch dimension to kernel centers.
        kc = self._kernel_centers[tf.newaxis, ...]

        return x - kc

    def _gaussian1d(self, x):
        # Add batch dimension to variance.
        var = self._compute_variance()
        var = var[tf.newaxis, ...]

        norm_coeff = 1.0 / tf.sqrt(2.0 * pi * var)
        exponent = -0.5 * x * x / var
        gaussians = norm_coeff * tf.exp(exponent)

        return gaussians

    def _demean(self, x):
        return x - tf.reduce_mean(x, axis=0, keepdims=True)

    def _create_weight_logits(self):
        self._weight_logits = tf.Variable(
            Constant(1.0)(shape=[self._n_timesteps, self._num_kernels]),
            trainable=True,
            name="weight_logits",
        )

    def _create_kernel_centers(self):
        self._kernel_centers = tf.Variable(
            RandomUniform()(shape=[self._n_timesteps, self._num_kernels, self._n_dims]),
            trainable=True,
            name="kernel_centers",
        )

    def _create_log_variance(self):
        self._log_variance = tf.Variable(
            Constant(1.0)(shape=[self._n_timesteps, self._num_kernels, self._n_dims]),
            trainable=True,
            name="log_variance",
        )

    @property
    def _n_dims(self):
        return self._input_shape[2]

    @property
    def _n_timesteps(self):
        return self._input_shape[1]

    @property
    def _batch_size(self):
        return self._input_shape[0]
    

class ProbabilityEstimatorCovarianceDiagonalizingKernel(tensorflow.keras.Model):
    def __init__(
        self,
        num_kernels=4,
        weight_logits_softmax_gain=3.5,
        max_log_variance_magnitude=3.5,
        eps=1e-7,
        name="probability_estimator",
        *args,
        **kwargs
    ):
        self._num_kernels = num_kernels
        self._weight_logits_softmax_gain = weight_logits_softmax_gain
        self._max_log_variance_magnitude = max_log_variance_magnitude
        self._eps = eps

        super(ProbabilityEstimatorCovarianceDiagonalizingKernel, self).__init__(name=name)

    def get_config(self):
        config = super(ProbabilityEstimatorCovarianceDiagonalizingKernel, self).get_config()
        config.update({"num_kernels": self._num_kernels, "eps": self._eps})
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape

        self._create_kernel_centers()
        self._create_weight_logits()
        self._create_log_variance()

        self.layer_losses = []
        
        if self._n_dims > 1:
            self._create_cov_diag_map_parametrization()

    def call(self, x, training=False):
        p = self._estimate_probabilities(x)

        if training:
            h = self._compute_entropy(p)
            self._add_regularizations(h)

        return p

    def _add_regularizations(self, h):
        h_mean = tf.reduce_mean(h)
        
        self.layer_losses.clear()
        self.layer_losses.append(h_mean)

    def _compute_entropy(self, p):
        p = tf.maximum(p, self._eps)
        h = -tf.reduce_mean(tf.math.log(p), axis=0)

        return h

    def _estimate_probabilities(self, x):
        x = self._subtract_kernel_centers(x)
        if self._n_dims > 1:
            x = self._apply_covariance_diagonalizing_transformations(x)

        gaussian_probability_estimations = self._gaussian1d(x)

        p = self._mix_gaussian_probability_estimations(gaussian_probability_estimations)
        return p

    def _apply_covariance_diagonalizing_transformations(self, x):
        # Transform x to the covariance diagonalizing coordinate system.
        transformations = self._compute_diagonalizing_transformations()
        x_transformed = tf.einsum("bnki, nkij->bnkj", x, transformations)

        return x_transformed

    def _mix_gaussian_probability_estimations(self, gaussian_probability_estimations):
        # Since each kernel has different transformation, probabilities
        # for two axis should be converted to joint probability
        # before merging kernels.
        weights = self._compute_gaussian_kernel_weights()

        lgpe = tf.math.log(gaussian_probability_estimations + self._eps)
        lgpe = tf.reduce_sum(lgpe, axis=-1)
        p_kernels = tf.exp(lgpe)
        p = tf.reduce_sum(p_kernels * weights[tf.newaxis, ...], axis=-1)

        return p

    def _compute_diagonalizing_transformations(self):
        ut = self._vector_to_upper_triangular_matrix(self._cov_diag_map_parametrization)

        antisym_matrix = 0.5 * (tf.linalg.matrix_transpose(ut) - ut)
        q = self._cayley_transformation(antisym_matrix)
        return q

    def _compute_gaussian_kernel_weights(self):
        clipped_weight_logits = tf.clip_by_value(self._weight_logits, 
                                                 -self._weight_logits_softmax_gain, 
                                                 self._weight_logits_softmax_gain)
        
        return tf.nn.softmax(clipped_weight_logits, axis=-1)

    def _compute_variance(self):
        clipped_log_var = tf.clip_by_value(self._log_variance, -self._max_log_variance_magnitude, self._max_log_variance_magnitude)
        var = tf.exp(clipped_log_var)
        return var

    def _subtract_kernel_centers(self, x):
        # Add kernel dimension to x.
        x = x[:, :, tf.newaxis, :]

        # Add batch dimension to kernel centers.
        kc = self._kernel_centers[tf.newaxis, ...]

        return x - kc

    def _gaussian1d(self, x):
        # Add batch dimension to variance.
        var = self._compute_variance()
        var = var[tf.newaxis, ...]

        norm_coeff = 1.0 / tf.sqrt(2.0 * pi * var)
        exponent = -0.5 * x * x / var
        gaussians = norm_coeff * tf.exp(exponent)

        return gaussians

    def _cayley_transformation(self, a):
        id = tf.eye(self._n_dims)
        id = id[tf.newaxis, tf.newaxis, :, :]
        q = tf.matmul(id - a, tf.linalg.inv(id + a))
        return q

    def _vector_to_upper_triangular_matrix(self, v):
        zero_pad = tf.zeros([self._n_timesteps, self._num_kernels, self._n_dims])

        a_flattened = tf.concat([v, v[..., ::-1], zero_pad], axis=2)
        a = tf.reshape(
            a_flattened,
            shape=[self._n_timesteps, self._num_kernels, self._n_dims, self._n_dims],
        )
        idxs = tf.range(0, self._n_dims)
        mask = tf.where((idxs[:, None] < idxs[None, :]), 1.0, 0.0)
        return mask[tf.newaxis, tf.newaxis, :, :] * a

    def _demean(self, x):
        return x - tf.reduce_mean(x, axis=0, keepdims=True)

    def _create_weight_logits(self):
        self._weight_logits = tf.Variable(
            Constant(1.0)(shape=[self._n_timesteps, self._num_kernels]),
            trainable=True,
            name="weight_logits",
        )

    def _create_kernel_centers(self):
        self._kernel_centers = tf.Variable(
            RandomUniform()(shape=[self._n_timesteps, self._num_kernels, self._n_dims]),
            trainable=True,
            name="kernel_centers",
        )

    def _create_log_variance(self):
        self._log_variance = tf.Variable(
            Constant(1.0)(shape=[self._n_timesteps, self._num_kernels, self._n_dims]),
            trainable=True,
            name="log_variance",
        )

    def _create_cov_diag_map_parametrization(self):
        d = self._n_dims
        n_elements = d * (d - 1) // 2
        self._cov_diag_map_parametrization = tf.Variable(
            Constant(1.0)(shape=[self._n_timesteps, self._num_kernels, n_elements]),
            trainable=True,
            name="cov_diag_map_parametrization",
        )

    @property
    def _n_dims(self):
        return self._input_shape[2]

    @property
    def _n_timesteps(self):
        return self._input_shape[1]

    @property
    def _batch_size(self):
        return self._input_shape[0]