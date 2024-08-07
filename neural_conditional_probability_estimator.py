import tensorflow
import tensorflow as tf
from math import pi
from tensorflow.keras.initializers import GlorotUniform, Zeros


class NeuralConditionalProbabilityEstimator(tensorflow.keras.Model):
    def __init__(
        self,
        hidden_layer_size=512,
        num_kernels=4,
        max_log_variance_magnitude=3.5,
        weight_logits_softmax_gain=2.5,
        given_and_estimated_on_same_space=True,
        mlp_l2_reg_coeff=1e-4,
        eps=1e-6,
        input_mask=None,
        entropy_regularization_mask=None,
        name="neural_conditional_probability_estimator",
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

        super(NeuralConditionalProbabilityEstimator, self).__init__(name=name)

    def get_config(self):
        config = super(NeuralConditionalProbabilityEstimator, self).get_config()
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape

        self._create_w_mlp_variables()
        self._create_mu_mlp_variables()
        self._create_var_mlp_variables()
        self._create_batch_mask()
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

    def _matrix_l2_norm(self, m):
        return tf.sqrt(tf.reduce_sum(tf.square(m)))

    def _estimate_conditional_probability(self, x_given, x_estimated):
        w, mu, var = self._estimate_gaussian_params(x_given)
        x_estimated = self._subtract_mean(x_estimated, mu)

        gaussian_probability_estimations = self._gaussian1d(x_estimated, var)

        p_cond = self._mix_gaussian_probability_estimations(
            w, gaussian_probability_estimations
        )

        p_cond = self._reshape_to_pairwise_conditional_probability_representation(
            p_cond
        )

        return p_cond

    def _estimate_gaussian_params(self, x):
        x_masked = self._apply_batch_mask(x)

        w = self._w_mlp(x_masked)
        mu = self._mu_mlp(x_masked)
        var = self._var_mlp(x_masked)

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
        x = tf.expand_dims(x, axis=2)

        return x - mu

    def _w_mlp(self, x):
        x = self._flatten(x)
        h = self._layer_lrelu(x, self._weights_l1_w, self._weights_l1_b)
        h = self._layer_lrelu(h, self._weights_l2_w, self._weights_l2_b)
        y = self._layer_linear(h, self._weights_l3_w, self._weights_l3_b)

        output_shape = [
            self._batch_size,
            self._n_estimated_timesteps,
            self._num_kernels,
        ]
        y = tf.reshape(y, shape=output_shape)
        w = tf.nn.softmax(self._weight_logits_softmax_gain * tf.tanh(y), axis=-1)

        return w

    def _mu_mlp(self, x):
        x = self._flatten(x)
        h = self._layer_lrelu(x, self._mu_l1_w, self._mu_l1_b)
        h = self._layer_lrelu(h, self._mu_l2_w, self._mu_l2_b)
        y = self._layer_linear(h, self._mu_l3_w, self._mu_l3_b)

        output_shape = [
            self._batch_size,
            self._n_estimated_timesteps,
            self._num_kernels,
            self._n_estimated_dims,
        ]

        y = tf.reshape(y, shape=output_shape)
        return y

    def _var_mlp(self, x):
        x = self._flatten(x)
        h = self._layer_lrelu(x, self._var_l1_w, self._var_l1_b)
        h = self._layer_lrelu(h, self._var_l2_w, self._var_l2_b)
        y = self._layer_linear(h, self._var_l3_w, self._var_l3_b)

        output_shape = [
            self._batch_size,
            self._n_estimated_timesteps,
            self._num_kernels,
            self._n_estimated_dims,
        ]
        y = tf.reshape(y, shape=output_shape)
        var = tf.exp(self._max_log_variance_magnitude * tf.tanh(y))

        return var

    def _entropy(self, p):
        p = tf.maximum(p, self._eps)
        h = -tf.reduce_mean(tf.math.log(p), axis=0)

        return h

    def _reshape_to_pairwise_conditional_probability_representation(self, p_cond):
        shape = [
            self._batch_size // self._n_given_timesteps,
            self._n_given_timesteps,
            self._n_estimated_timesteps,
        ]

        p_cond = tf.reshape(p_cond, shape=shape)
        return p_cond

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

    def _apply_batch_mask(self, x):
        x_masked = x * self._batch_mask[..., tf.newaxis]
        return x_masked

    def _flatten(self, x):
        x_shape = tf.shape(x)
        x = tf.reshape(x, [x_shape[0], x_shape[1] * x_shape[2]])
        return x

    def _create_w_mlp_variables(self):
        in_size = self._n_given_timesteps * self._n_given_dims
        out_size = self._n_estimated_timesteps * self._num_kernels

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
        in_size = self._n_given_timesteps * self._n_given_dims
        out_size = (
            self._n_estimated_timesteps * self._num_kernels * self._n_estimated_dims
        )

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
        in_size = self._n_given_timesteps * self._n_given_dims
        out_size = (
            self._n_estimated_timesteps * self._num_kernels * self._n_estimated_dims
        )

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

    def _layer_lrelu(self, x, w, b):
        y = tf.matmul(x, w) + b
        y = tf.nn.leaky_relu(y)

        return y

    def _layer_linear(self, x, w, b):
        y = tf.matmul(x, w) + b

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

    def _create_batch_mask(self):
        if self._input_mask == None:
            self._input_mask = tf.eye(self._n_given_timesteps, dtype=tf.float32)

        batch_mask = tf.expand_dims(self._input_mask, axis=0)
        batch_mask = tf.repeat(
            batch_mask, axis=0, repeats=self._batch_size // self._n_given_timesteps
        )
        batch_mask = tf.reshape(batch_mask, [self._batch_size, self._n_given_timesteps])
        self._batch_mask = batch_mask

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