import tensorflow
import tensorflow as tf
from math import pi
from tensorflow.keras.initializers import GlorotUniform, Zeros


class NeuralConditionalProbabilityEstimatorMultiple(tensorflow.keras.Model):
    def __init__(
        self,
        num_kernels=4,
        max_log_variance_magnitude=3.5,
        weight_logits_softmax_gain=2.5,
        given_and_estimated_on_same_space=True,
        mlp_l2_reg_coeff=1e-4,
        eps=1e-6,
        input_mask=None,
        entropy_regularization_mask=None,
        hidden_layer_size=None,
        name="neural_conditional_probability_estimator_multiple",
        *args,
        **kwargs
    ):
        self._hidden_layer_size = hidden_layer_size
        self._num_kernels = num_kernels
        self._max_log_variance_magnitude = max_log_variance_magnitude
        self._weight_logits_softmax_gain = weight_logits_softmax_gain
        self._given_and_estimated_on_same_space = given_and_estimated_on_same_space
        self._mlp_l2_reg_coeff = mlp_l2_reg_coeff
        self._eps = eps
        self._input_mask = input_mask
        self._entropy_regularization_mask = entropy_regularization_mask

        super(NeuralConditionalProbabilityEstimatorMultiple, self).__init__(name=name)

    def get_config(self):
        config = super(NeuralConditionalProbabilityEstimatorMultiple, self).get_config()
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape

        if self._hidden_layer_size is None:
            self._hidden_layer_size = 2 * self._n_estimated_timesteps * self._n_estimated_dims * self._num_kernels
        else: 
            if self._hidden_layer_size % (self._n_estimated_timesteps * self._n_estimated_dims * self._num_kernels) != 0:
                raise ValueError("Hidden layer size should be multiples of dims multiplied with number of kernels.")

        self._create_w_mlp_variables()
        self._create_mu_mlp_variables()
        self._create_var_mlp_variables()
        self._create_entropy_regularization_mask()

        self.layer_losses = []
        
    def call(self, x, training=False):
        x_given, x_estimated = x
        p_cond = self._estimate_conditional_probability(x_given, x_estimated)

        if training:
            h_cond = self._entropy(p_cond)
            self._add_regularizations(h_cond)
            

        return p_cond

    def _add_regularizations(self, h):
        l2_reg = self._compute_l2_reg()
        h_mean = tf.reduce_sum(h * self._entropy_regularization_mask)
        h_mean = h_mean / tf.reduce_sum(self._entropy_regularization_mask)
        
        self.layer_losses.clear()
        
        self.layer_losses.append(h_mean)
        self.layer_losses.append(l2_reg)

    def _compute_l2_reg(self):
        mu_w_l2 = (
            self._matrix_l2_norm(self._mu_l1_w)
            + self._matrix_l2_norm(self._mu_l2_w)
        )

        var_w_l2 = (
            self._matrix_l2_norm(self._var_l1_w)
            + self._matrix_l2_norm(self._var_l2_w)
        )

        weigths_w_l2 = (
            self._matrix_l2_norm(self._weights_l1_w)
            + self._matrix_l2_norm(self._weights_l2_w)
        )

        return self._mlp_l2_reg_coeff * (mu_w_l2 + var_w_l2 + weigths_w_l2)

    def _matrix_l2_norm(self, m):
        return tf.sqrt(tf.reduce_sum(tf.square(m)))

    def _estimate_conditional_probability(self, x_given, x_estimated):
        w, mu, var = self._estimate_gaussian_params(x_given)

        x_estimated = self._subtract_mean(x_estimated, mu)

        gaussian_probability_estimations = self._gaussian1d(x_estimated, var)

        p_cond = self._mix_gaussian_probability_estimations(
            w, gaussian_probability_estimations
        )

        return p_cond

    def _estimate_gaussian_params(self, x_given):
        w = self._w_mlp(x_given)
        mu = self._mu_mlp(x_given)
        var = self._var_mlp(x_given)

        return w, mu, var

    def _mix_gaussian_probability_estimations(
        self, weights, gaussian_probability_estimations
    ):
        log_p = tf.math.log(gaussian_probability_estimations + self._eps)

        # Sum dimensions.
        log_p = tf.reduce_sum(log_p, axis=-1)

        # Sum kernels.
        p_kernels = tf.exp(log_p)
        p = tf.reduce_sum(p_kernels * weights, axis=-1)

        return p

    def _subtract_mean(self, x, mu):
        # Add kernels axis to x.
        x = tf.expand_dims(x, axis=-2)

        # Add given axis to x.
        x = tf.expand_dims(x, axis=1)

        return x - mu

    def _w_mlp(self, x):
        h = self._layer_lrelu(x, self._weights_l1_w, self._weights_l1_b)
        y = self._layer_linear(h, self._weights_l2_w, self._weights_l2_b)

        output_shape = [
            self._batch_size,
            self._n_given_timesteps,
            self._n_estimated_timesteps,
            self._num_kernels
        ]
        y = tf.reshape(y, shape=output_shape)
        y = tf.clip_by_value(y, 
                             clip_value_min=-self._weight_logits_softmax_gain, 
                             clip_value_max=self._weight_logits_softmax_gain)
        
        w = tf.nn.softmax(y, axis=-1)

        return w

    def _mu_mlp(self, x):
        h = self._layer_lrelu(x, self._mu_l1_w, self._mu_l1_b)
        y = self._layer_linear(h, self._mu_l2_w, self._mu_l2_b)

        output_shape = [
            self._batch_size,
            self._n_given_timesteps,
            self._n_estimated_timesteps,
            self._num_kernels,
            self._n_estimated_dims,
        ]

        y = tf.reshape(y, shape=output_shape)
        return y

    def _var_mlp(self, x):
        h = self._layer_lrelu(x, self._var_l1_w, self._var_l1_b)
        y = self._layer_linear(h, self._var_l2_w, self._var_l2_b)

        output_shape = [
            self._batch_size,
            self._n_given_timesteps,
            self._n_estimated_timesteps,
            self._num_kernels,
            self._n_estimated_dims,
        ]
        y = tf.reshape(y, shape=output_shape)
        y = tf.clip_by_value(y, 
                             clip_value_min=-self._max_log_variance_magnitude, 
                             clip_value_max=self._max_log_variance_magnitude)
        var = tf.exp(y)

        return var

    def _entropy(self, p):
        p = tf.maximum(p, self._eps)
        h = -tf.reduce_mean(tf.math.log(p), axis=0)

        return h

    def _gaussian1d(self, x, var):
        c = 1.0 / tf.sqrt(2.0 * pi * var)
        gaussians = c * tf.exp(-0.5 * x * x / var)
        return gaussians

    def _get_conditional_entropy_mask(self):
        mask = tf.ones(shape=(self._n_given_timesteps, self._n_estimated_timesteps))

        if self._given_and_estimated_on_same_space:
            return mask - tf.eye(self._n_given_timesteps)
        else:
            return mask

    def _create_w_mlp_variables(self):
        in_size = self._n_given_dims
        out_size = self._n_estimated_timesteps * self._num_kernels

        self._weights_l1_w = self._create_trainable_variable(
            shape=[self._n_given_timesteps, in_size, self._hidden_layer_size],
            name="weights_l1_w",
        )

        self._weights_l1_b = self._create_trainable_variable(
            shape=[1, self._n_given_timesteps, self._hidden_layer_size],
            name="weights_l1_b",
            initializer="ze",
        )

        self._weights_l2_w = self._create_trainable_variable(
            shape=[
                self._n_given_timesteps,
                self._hidden_layer_size,
                out_size,
            ],
            name="weights_l2_w",
        )

        self._weights_l2_b = self._create_trainable_variable(
            shape=[1, self._n_given_timesteps, out_size],
            name="weights_l2_b",
            initializer="ze",
        )

    def _create_mu_mlp_variables(self):
        in_size = self._n_given_dims
        out_size = (
            self._n_estimated_timesteps * self._num_kernels * self._n_estimated_dims
        )

        self._mu_l1_w = self._create_trainable_variable(
            shape=[self._n_given_timesteps, in_size, self._hidden_layer_size],
            name="mu_l1_w",
        )

        self._mu_l1_b = self._create_trainable_variable(
            shape=[1, self._n_given_timesteps, self._hidden_layer_size],
            name="mu_l1_b",
            initializer="ze",
        )

        self._mu_l2_w = self._create_trainable_variable(
            shape=[self._n_given_timesteps, self._hidden_layer_size, out_size],
            name="mu_l2_w",
        )

        self._mu_l2_b = self._create_trainable_variable(
            shape=[1, self._n_given_timesteps, out_size],
            name="mu_l2_b",
            initializer="ze",
        )

    def _create_var_mlp_variables(self):
        in_size = self._n_given_dims
        out_size = (
            self._n_estimated_timesteps * self._num_kernels * self._n_estimated_dims
        )

        self._var_l1_w = self._create_trainable_variable(
            shape=[self._n_given_timesteps, in_size, self._hidden_layer_size],
            name="var_l1_w",
        )

        self._var_l1_b = self._create_trainable_variable(
            shape=[1, self._n_given_timesteps, self._hidden_layer_size],
            name="var_l1_b",
            initializer="ze",
        )

        self._var_l2_w = self._create_trainable_variable(
            shape=[self._n_given_timesteps, self._hidden_layer_size, out_size],
            name="var_l3_w",
        )

        self._var_l2_b = self._create_trainable_variable(
            shape=[1, self._n_given_timesteps, out_size],
            name="var_l3_b",
            initializer="ze",
        )

    def _layer_lrelu(self, x, w, b):
        y = self._layer_linear(x, w, b)
        y = tf.nn.leaky_relu(y)

        return y

    def _layer_linear(self, x, w, b):
        y = tf.einsum("bac, acd->bad", x, w) + b

        return y

    def _create_trainable_variable(self, shape, name, initializer="gu"):
        if initializer == "gu":
            initializer = GlorotUniform()
        elif initializer == "ze":
            initializer = Zeros()
        else:
            print("Error")

        var = tf.Variable(initializer(shape), trainable=True, name=name)
        return var

    def _create_entropy_regularization_mask(self):
        if self._entropy_regularization_mask == None:
            if self._given_and_estimated_on_same_space:
                self._entropy_regularization_mask = 1.0 - tf.eye(
                    self._n_given_timesteps,
                    self._n_estimated_timesteps,
                    dtype=tf.float32,
                )
            else:
                self._entropy_regularization_mask = tf.ones(
                    shape=[self._n_given_timesteps, self._n_estimated_timesteps],
                    dtype=tf.float32,
                )

    @property
    def _n_estimated_dims(self):
        return self._input_shape[1][2]

    @property
    def _n_estimated_timesteps(self):
        return self._input_shape[1][1]

    @property
    def _n_given_dims(self):
        return self._input_shape[0][2]

    @property
    def _n_given_timesteps(self):
        return self._input_shape[0][1]

    @property
    def _batch_size(self):
        return self._input_shape[0][0]