import tensorflow
import tensorflow as tf
from math import pi
from tensorflow.keras.initializers import GlorotUniform, Zeros


class NeuralEntropyEstimator(tensorflow.keras.models.Model):
    def __init__(
        self,
        num_kernels=4,
        max_log_variance_magnitude=15.0,
        weight_logits_softmax_gain=10.0,
        mlp_l2_reg_coeff=1e-4,
        eps=1e-7,
        hidden_layer_size=None,
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

        if self._hidden_layer_size is None:
            self._hidden_layer_size = 2 * self._n_dims * self._num_kernels
        else: 
            if self._hidden_layer_size % (self._n_dims * self._num_kernels) != 0:
                raise ValueError("Hidden layer size should be multiples of dims multiplied with number of kernels.")
            
        self.layer_losses = []

        self._create_w_mlp_variables()
        self._create_mu_mlp_variables()
        self._create_var_mlp_variables()

    def call(self, x, training=False):
        p = self._estimate_conditional_probability(x)
        
        h_estimation = self._entropy(p)
        self._clear_layer_losses()
        
        if training:
            self._add_layer_loss(self._compute_entropy_regularization(h_estimation))
            self._add_layer_loss(self._compute_l2_regularization())

        return h_estimation
    
    def _clear_layer_losses(self):
        self.layer_losses = []

    def _add_layer_loss(self, loss):
        self.layer_losses.append(loss)
        
    def _compute_entropy_regularization(self, h):
        h_mean = tf.reduce_mean(h)
        return h_mean
    
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
        w1 = self._mask_layer_weights_for_conditional_probability_estimation(
            self._weights_l1_w
        )
        w2 = self._mask_layer_weights_for_conditional_probability_estimation(
            self._weights_l2_w
        )
        w3 = self._mask_layer_weights_for_conditional_probability_estimation(
            self._weights_l3_w
        )

        h = self._layer_lrelu(x, w1, self._weights_l1_b)
        h = self._layer_lrelu(h, w2, self._weights_l2_b)
        y = self._layer_linear(h, w3, self._weights_l3_b)

        output_shape = [
            self._batch_size,
            self._n_estimators,
            self._n_dims,
            self._num_kernels
        ]
        y = tf.reshape(y, shape=output_shape)
        y = tf.roll(y, shift=1, axis=-2)
        w = tf.nn.softmax(self._weight_logits_softmax_gain * tf.tanh(y), axis=-1)

        return w

    def _mu_mlp(self, x):
        w1 = self._mask_layer_weights_for_conditional_probability_estimation(self._mu_l1_w)
        w2 = self._mask_layer_weights_for_conditional_probability_estimation(self._mu_l2_w)
        w3 = self._mask_layer_weights_for_conditional_probability_estimation(self._mu_l3_w)

        h = self._layer_lrelu(x, w1, self._mu_l1_b)
        h = self._layer_lrelu(h, w2, self._mu_l2_b)
        y = self._layer_linear(h, w3, self._mu_l3_b)

        output_shape = [
            self._batch_size,
            self._n_estimators,
            self._n_dims,
            self._num_kernels
        ]

        y = tf.reshape(y, shape=output_shape)
        y = tf.roll(y, shift=1, axis=-2)
        return y

    def _var_mlp(self, x):
        w1 = self._mask_layer_weights_for_conditional_probability_estimation(self._var_l1_w)
        w2 = self._mask_layer_weights_for_conditional_probability_estimation(self._var_l2_w)
        w3 = self._mask_layer_weights_for_conditional_probability_estimation(self._var_l3_w)

        h = self._layer_lrelu(x, w1, self._var_l1_b)
        h = self._layer_lrelu(h, w2, self._var_l2_b)
        y = self._layer_linear(h, w3, self._var_l3_b)

        output_shape = [
            self._batch_size,
            self._n_estimators,
            self._n_dims,
            self._num_kernels
        ]
        y = tf.reshape(y, shape=output_shape)
        y = tf.roll(y, shift=1, axis=-2)
        var = tf.exp(self._max_log_variance_magnitude * tf.tanh(y))

        return var

    def _mask_layer_weights_for_conditional_probability_estimation(self, w):
        w_shape = tf.shape(w)
        layer_in_dim = w_shape[1]
        layer_out_dim = w_shape[2]
        num_cells = self._input_shape[-1]
        
        in_node_cell_idx = (tf.range(0, layer_in_dim, dtype=tf.int32) * num_cells) // layer_in_dim
        out_node_cell_idx = (tf.range(0, layer_out_dim, dtype=tf.int32) * num_cells) // layer_out_dim
        
        # Last path is used for estimating the first element so no information 
        #is required.
        mask = tf.where(tf.logical_and((in_node_cell_idx[:, tf.newaxis] <= out_node_cell_idx[tf.newaxis, :]), 
                                       out_node_cell_idx[tf.newaxis, :] < (num_cells-1)), 
                        1.0, 0.0)

        return w * mask
    
    def _gaussian_kernel(self, x, mu, var):
        # Add kernels dimension to input.
        x = tf.expand_dims(x, axis=-1)
        
        # Compute normalization constant.
        c = 1.0 / tf.sqrt(2.0 * pi * var)
        
        x = x - mu
        gaussians = c * tf.exp(-0.5 * x * x / var)
        return gaussians
    
    def _covariance(self, x):
        x = x - tf.reduce_mean(x, axis=0)
        cov = tf.matmul(x, x, transpose_a=True)
        return cov
        
    def _entropy(self, p):
        h = -tf.reduce_mean(tf.math.log(p), axis=0)

        return h
        
    def _gaussian_entropy_1d(self, variance):
        entropy = 0.5 * (1.0 + tf.math.log(2.0 * pi * variance + self._eps))
        return entropy

    def _matrix_l2_norm(self, m):
        return tf.sqrt(tf.reduce_sum(tf.square(m)))
    
    def _layer_lrelu(self, x, w, b):
        y = self._layer_linear(x, w, b)
        y = tf.nn.leaky_relu(y)

        return y

    def _layer_linear(self, x, w, b):
        y = tf.einsum("bnd, nde->bne", x, w) + b
        return y
        
    def _create_w_mlp_variables(self):
        in_size = self._n_dims
        out_size = self._n_dims * self._num_kernels

        self._weights_l1_w = self._create_trainable_variable(
            shape=[self._n_estimators, in_size, self._hidden_layer_size],
            name="weights_l1_w",
        )

        self._weights_l1_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, self._hidden_layer_size],
            name="weights_l1_b",
            initializer="ze",
        )

        self._weights_l2_w = self._create_trainable_variable(
            shape=[
                self._n_estimators,
                self._hidden_layer_size,
                self._hidden_layer_size,
            ],
            name="weights_l2_w",
        )

        self._weights_l2_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, self._hidden_layer_size],
            name="weights_l2_b",
            initializer="ze",
        )

        self._weights_l3_w = self._create_trainable_variable(
            shape=[
                self._n_estimators,
                self._hidden_layer_size,
                out_size,
            ],
            name="weights_l3_w",
        )

        self._weights_l3_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, out_size],
            name="weights_l3_b",
            initializer="ze",
        )

    def _create_mu_mlp_variables(self):
        in_size = self._n_dims
        out_size = self._n_dims * self._num_kernels

        self._mu_l1_w = self._create_trainable_variable(
            shape=[self._n_estimators, in_size, self._hidden_layer_size],
            name="mu_l1_w",
        )

        self._mu_l1_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, self._hidden_layer_size],
            name="mu_l1_b",
            initializer="ze",
        )

        self._mu_l2_w = self._create_trainable_variable(
            shape=[self._n_estimators, self._hidden_layer_size, 
                   self._hidden_layer_size],
            name="mu_l2_w",
        )

        self._mu_l2_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, self._hidden_layer_size],
            name="mu_l2_b",
            initializer="ze",
        )

        self._mu_l3_w = self._create_trainable_variable(
            shape=[self._n_estimators, self._hidden_layer_size, out_size],
            name="mu_l3_w",
        )

        self._mu_l3_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, out_size],
            name="mu_l3_b",
            initializer="ze",
        )

    def _create_var_mlp_variables(self):
        in_size = self._n_dims
        out_size = self._n_dims * self._num_kernels

        self._var_l1_w = self._create_trainable_variable(
            shape=[self._n_estimators, in_size, self._hidden_layer_size],
            name="var_l1_w",
        )

        self._var_l1_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, self._hidden_layer_size],
            name="var_l1_b",
            initializer="ze",
        )

        self._var_l2_w = self._create_trainable_variable(
            shape=[self._n_estimators, self._hidden_layer_size, self._hidden_layer_size],
            name="var_l2_w",
        )

        self._var_l2_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, self._hidden_layer_size],
            name="var_l2_b",
            initializer="ze",
        )

        self._var_l3_w = self._create_trainable_variable(
            shape=[self._n_estimators, self._hidden_layer_size, out_size],
            name="var_l3_w",
        )

        self._var_l3_b = self._create_trainable_variable(
            shape=[1, self._n_estimators, out_size],
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
        return self._input_shape[2]

    @property
    def _n_estimators(self):
        return self._input_shape[1]
    
    @property
    def _batch_size(self):
        return self._input_shape[0]
