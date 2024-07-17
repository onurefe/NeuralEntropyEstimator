import tensorflow
import tensorflow as tf
from math import pi
from tensorflow.keras.initializers import GlorotUniform, Zeros


class NeuralEntropyEstimator(tensorflow.keras.Model):
    def __init__(
        self,
        hidden_layer_size=528,
        num_kernels=4,
        max_log_variance_magnitude=15.0,
        weight_logits_softmax_gain=10.0,
        mlp_l2_reg_coeff=1e-4,
        eps=1e-7,
        name="neural_entropy_estimator",
        *args,
        **kwargs
    ):
        self._hidden_layer_size = hidden_layer_size
        self._num_kernels = num_kernels
        self._max_log_variance_magnitude = max_log_variance_magnitude
        self._weight_logits_softmax_gain = weight_logits_softmax_gain
        self._mlp_l2_reg_coeff = mlp_l2_reg_coeff
        self._eps = eps

        super(NeuralEntropyEstimator, self).__init__(name=name)

    def get_config(self):
        config = super(NeuralEntropyEstimator, self).get_config()
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape

        self._create_covariance_diagonalizing_map()
        self._create_w_mlp_variables()
        self._create_mu_mlp_variables()
        self._create_var_mlp_variables()

    def call(self, x, rank=None, training=False):
        if rank is None:
            rank = self._n_dims
        else:
            rank = self._clip_rank(rank)
                
        x = self._flatten(x)
        y = self._map_to_conditional_probability_input_space(x)
        p = self._estimate_conditional_probability(y)
        h_estimation = self._entropy(p)

        if training:
            self.add_loss(self._compute_top_k_subspace_entropy_regularization(h_estimation, rank))
            self.add_loss(self._compute_top_k_subspace_covariance_diagonalization_regularization(x, rank))
            self.add_loss(self._compute_top_k_subspace_unitarity_regularization(rank))
            self.add_loss(self._compute_top_k_subspace_entropy_maximization_regularization(x, rank))
            self.add_loss(self._compute_l2_regularization())

        return h_estimation
    
    def _compute_top_k_subspace_entropy_regularization(self, h, k):
        mask = self._vector_subspace_mask(k)
        entropy_per_dim = tf.reduce_sum(h * mask) / tf.reduce_sum(mask)
        
        return entropy_per_dim
    
    def _compute_top_k_subspace_covariance_diagonalization_regularization(self, x, k):
        normalized_map = self._normalize_columns(self._covariance_diagonalizing_map)
        x_mapped = tf.matmul(x, normalized_map)
        
        bs = tf.shape(x)[0]
        covariance = tf.matmul(x_mapped, x_mapped, transpose_a=True) / tf.cast(bs - 1, tf.float32)
        
        off_diagonal_mask =  (tf.ones_like(covariance) - tf.eye(self._n_dims))
        off_diagonal_elements = self._matrix_subspace_mask(k) * covariance * off_diagonal_mask
        covariance_loss = tf.norm(off_diagonal_elements, ord='fro', axis=(0,1))
        covariance_loss_per_dim = covariance_loss / tf.cast(k, tf.float32)
        
        return covariance_loss_per_dim

    def _compute_top_k_subspace_unitarity_regularization(self, k):
        gram_matrix = tf.matmul(self._covariance_diagonalizing_map, 
                                self._covariance_diagonalizing_map, 
                                transpose_a=True)
        
        identity_matrix = tf.eye(self._n_dims)
        unitarity_loss = tf.norm(self._matrix_subspace_mask(k) * (gram_matrix - identity_matrix), 
                                 ord='fro', 
                                 axis=(0,1))
        unitarity_loss_per_dim = unitarity_loss / tf.cast(k, tf.float32)
        
        return unitarity_loss_per_dim

    def _compute_top_k_subspace_entropy_maximization_regularization(self, x, k):
        normalized_map = self._normalize_columns(self._covariance_diagonalizing_map)
        x_mapped = tf.matmul(x, normalized_map)
        variance = tf.math.reduce_variance(x_mapped, axis=0)
        h = self._gaussian_entropy_1d(variance)
        mask = self._vector_subspace_mask(k)
        h_mean = tf.reduce_sum(h * mask) / tf.reduce_sum(mask)
        return -h_mean
    
    def _compute_l2_regularization(self):
        mu_w_l2 = (
            self._matrix_l2_norm(self._mu_l1_w)
            + self._matrix_l2_norm(self._mu_l2_w)
            + self._matrix_l2_norm(self._mu_l3_w)
        )

        var_w_l2 = (
            self._matrix_l2_norm(self._var_l1_w)
            + self._matrix_l2_norm(self._var_l2_w)
            + self._matrix_l2_norm(self._var_l3_w)
        )

        weigths_w_l2 = (
            self._matrix_l2_norm(self._weights_l1_w)
            + self._matrix_l2_norm(self._weights_l2_w)
            + self._matrix_l2_norm(self._weights_l3_w)
        )

        return self._mlp_l2_reg_coeff * (mu_w_l2 + var_w_l2 + weigths_w_l2)
    
    def _estimate_conditional_probability(self, x):
        w, mu, var = self._estimate_gaussian_params(x)
        
        gaussian_probability_estimations = self._gaussian_kernel(x, mu, var)

        p_cond = self._mix_gaussian_probability_estimations(
            w, gaussian_probability_estimations
        )

        return p_cond

    def _estimate_gaussian_params(self, x):
        w = self._w_mlp(x)
        mu = self._mu_mlp(x)
        var = self._var_mlp(x)

        return w, mu, var

    def _mix_gaussian_probability_estimations(
        self, weights, gaussian_probability_estimations
    ):
        log_p = tf.math.log(gaussian_probability_estimations + self._eps)

        # Sum kernels.
        p_kernels = tf.exp(log_p)
        p = tf.reduce_sum(p_kernels * weights, axis=-1)

        return p

    def _w_mlp(self, x):
        w1 = self._mask_weights_for_conditional_probability_estimation(
            self._weights_l1_w
        )
        w2 = self._mask_weights_for_conditional_probability_estimation(
            self._weights_l2_w
        )
        w3 = self._mask_weights_for_conditional_probability_estimation(
            self._weights_l3_w
        )

        h = self._layer_lrelu(x, w1, self._weights_l1_b)
        h = self._layer_lrelu(h, w2, self._weights_l2_b)
        y = self._layer_linear(h, w3, self._weights_l3_b)

        output_shape = [
            self._batch_size,
            self._n_dims,
            self._num_kernels
        ]
        y = tf.reshape(y, shape=output_shape)
        w = tf.nn.softmax(self._weight_logits_softmax_gain * tf.tanh(y), axis=-1)

        return w

    def _mu_mlp(self, x):
        w1 = self._mask_weights_for_conditional_probability_estimation(self._mu_l1_w)
        w2 = self._mask_weights_for_conditional_probability_estimation(self._mu_l2_w)
        w3 = self._mask_weights_for_conditional_probability_estimation(self._mu_l3_w)

        h = self._layer_lrelu(x, w1, self._mu_l1_b)
        h = self._layer_lrelu(h, w2, self._mu_l2_b)
        y = self._layer_linear(h, w3, self._mu_l3_b)

        output_shape = [
            self._batch_size,
            self._n_dims,
            self._num_kernels
        ]

        y = tf.reshape(y, shape=output_shape)
        return y

    def _var_mlp(self, x):
        w1 = self._mask_weights_for_conditional_probability_estimation(self._var_l1_w)
        w2 = self._mask_weights_for_conditional_probability_estimation(self._var_l2_w)
        w3 = self._mask_weights_for_conditional_probability_estimation(self._var_l3_w)

        h = self._layer_lrelu(x, w1, self._var_l1_b)
        h = self._layer_lrelu(h, w2, self._var_l2_b)
        y = self._layer_linear(h, w3, self._var_l3_b)

        output_shape = [
            self._batch_size,
            self._n_dims,
            self._num_kernels
        ]
        y = tf.reshape(y, shape=output_shape)
        var = tf.exp(self._max_log_variance_magnitude * tf.tanh(y))

        return var

    def _mask_weights_for_conditional_probability_estimation(self, w):
        w_shape = tf.shape(w)
        in_dim = w_shape[0]
        out_dim = w_shape[1]
        
        in_idxs = tf.range(1, in_dim + 1, dtype=tf.float32)
        out_idxs = tf.range(1, out_dim + 1, dtype=tf.float32)
        in_idxs = in_idxs[:, tf.newaxis]
        out_idxs = out_idxs[tf.newaxis, :]

        dim_ratio = tf.cast(out_dim, tf.float32) / tf.cast(in_dim, tf.float32)
        mask = tf.where((tf.math.ceil(in_idxs * dim_ratio) < out_idxs), 1.0, 0.0)

        return w * mask

    def _gaussian_kernel(self, x, mu, var):
        # Add kernels dimension to input.
        x = tf.expand_dims(x, axis=-1)
        
        # Compute normalization constant.
        c = 1.0 / tf.sqrt(2.0 * pi * var)
        
        x = x - mu
        gaussians = c * tf.exp(-0.5 * x * x / var)
        return gaussians
    
    def _matrix_subspace_mask(self, rank):
        mask1d = self._vector_subspace_mask(rank)
        mask2d = mask1d[:, tf.newaxis] * mask1d[tf.newaxis, :]
        return mask2d
    
    def _vector_subspace_mask(self, rank):
        rankf = tf.cast(rank, tf.float32)
        component_idxs = tf.range(0, self._n_dims, dtype=tf.float32)
        mask1d = self._soft_thresholding_window(component_idxs, rankf)
        return mask1d
    
    def _soft_thresholding_window(self, x, threshold, gain=5, exp_clamp=50):
        exponent = gain * (x - threshold)
        exponent = tf.clip_by_value(
            exponent, clip_value_min=-exp_clamp, clip_value_max=exp_clamp
        )
        return 1.0 / (tf.exp(exponent) + 1.0)
    
    def _map_to_conditional_probability_input_space(self, x):
        normalized_map = self._normalize_columns(self._covariance_diagonalizing_map)
        return tf.matmul(x, tf.stop_gradient(normalized_map))
        
    def _entropy(self, p):
        h = -tf.reduce_mean(tf.math.log(p), axis=0)

        return h
    
    def _flatten(self, x):
        input_shape = tf.shape(x)
        flattened_dim = tf.reduce_prod(input_shape[1:])
        flattened_tensor = tf.reshape(x, (input_shape[0], flattened_dim))
        
        return flattened_tensor
    
    def _normalize_columns(self, matrix):
        column_norms = tf.sqrt(tf.reduce_sum(tf.square(matrix), axis=0, keepdims=True))
        normalized_matrix = matrix / (column_norms + self._eps)
        return normalized_matrix

    def _clip_rank(self, rank):
        rank = tf.minimum(tf.maximum(rank, 2), self._n_dims)
        return rank
    
    def _gaussian_entropy_1d(self, variance):
        entropy = 0.5 * (1.0 + tf.math.log(2.0 * pi * variance + self._eps))

        return entropy

    def _matrix_l2_norm(self, m):
        return tf.sqrt(tf.reduce_sum(tf.square(m)))
    
    def _layer_lrelu(self, x, w, b):
        y = tf.matmul(x, w) + b
        y = tf.nn.leaky_relu(y)

        return y

    def _layer_linear(self, x, w, b):
        y = tf.matmul(x, w) + b

        return y
    
    def _create_covariance_diagonalizing_map(self):
        self._covariance_diagonalizing_map = self._create_trainable_variable(
            shape=[self._n_dims, self._n_dims],
            name="covariance_diagonalizing_map",
        )
        
    def _create_w_mlp_variables(self):
        in_size = self._n_dims
        out_size = self._n_dims * self._num_kernels

        self._weights_l1_w = self._create_trainable_variable(
            shape=[in_size, self._hidden_layer_size],
            name="weights_l1_w",
        )

        self._weights_l1_b = self._create_trainable_variable(
            shape=[1, self._hidden_layer_size],
            name="weights_l1_b",
            initializer="ze",
        )

        self._weights_l2_w = self._create_trainable_variable(
            shape=[
                self._hidden_layer_size,
                self._hidden_layer_size,
            ],
            name="weights_l2_w",
        )

        self._weights_l2_b = self._create_trainable_variable(
            shape=[1, self._hidden_layer_size],
            name="weights_l2_b",
            initializer="ze",
        )

        self._weights_l3_w = self._create_trainable_variable(
            shape=[
                self._hidden_layer_size,
                out_size,
            ],
            name="weights_l3_w",
        )

        self._weights_l3_b = self._create_trainable_variable(
            shape=[1, out_size],
            name="weights_l3_b",
            initializer="ze",
        )

    def _create_mu_mlp_variables(self):
        in_size = self._n_dims
        out_size = self._n_dims * self._num_kernels

        self._mu_l1_w = self._create_trainable_variable(
            shape=[in_size, self._hidden_layer_size],
            name="mu_l1_w",
        )

        self._mu_l1_b = self._create_trainable_variable(
            shape=[1, self._hidden_layer_size],
            name="mu_l1_b",
            initializer="ze",
        )

        self._mu_l2_w = self._create_trainable_variable(
            shape=[self._hidden_layer_size, self._hidden_layer_size],
            name="mu_l2_w",
        )

        self._mu_l2_b = self._create_trainable_variable(
            shape=[1, self._hidden_layer_size],
            name="mu_l2_b",
            initializer="ze",
        )

        self._mu_l3_w = self._create_trainable_variable(
            shape=[self._hidden_layer_size, out_size],
            name="mu_l3_w",
        )

        self._mu_l3_b = self._create_trainable_variable(
            shape=[1, out_size],
            name="mu_l3_b",
            initializer="ze",
        )

    def _create_var_mlp_variables(self):
        in_size = self._n_dims
        out_size = self._n_dims * self._num_kernels

        self._var_l1_w = self._create_trainable_variable(
            shape=[in_size, self._hidden_layer_size],
            name="var_l1_w",
        )

        self._var_l1_b = self._create_trainable_variable(
            shape=[1, self._hidden_layer_size],
            name="var_l1_b",
            initializer="ze",
        )

        self._var_l2_w = self._create_trainable_variable(
            shape=[self._hidden_layer_size, self._hidden_layer_size],
            name="var_l2_w",
        )

        self._var_l2_b = self._create_trainable_variable(
            shape=[1, self._hidden_layer_size],
            name="var_l2_b",
            initializer="ze",
        )

        self._var_l3_w = self._create_trainable_variable(
            shape=[self._hidden_layer_size, out_size],
            name="var_l3_w",
        )

        self._var_l3_b = self._create_trainable_variable(
            shape=[1, out_size],
            name="var_l3_b",
            initializer="ze",
        )

    def _create_trainable_variable(self, shape, name, initializer="gu"):
        if initializer == "gu":
            initializer = GlorotUniform()
        elif initializer == "ze":
            initializer = Zeros()
        else:
            print("Error")

        var = tf.Variable(initializer(shape), trainable=True, name=name)
        return var

    @property
    def _n_dims(self):
        n_dims = tf.reduce_prod(self._input_shape[1:])
        
        return n_dims

    @property
    def _batch_size(self):
        return self._input_shape[0]
