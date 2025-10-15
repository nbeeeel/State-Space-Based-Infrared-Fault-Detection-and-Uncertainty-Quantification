import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging

class Float32Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape

class FourierThermalLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, lambda_physics=5.0, use_physics_loss=True, target_var=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.lambda_physics = lambda_physics
        self.use_physics_loss = use_physics_loss
        self.target_var = target_var
        self.alpha_subnetwork = tf.keras.Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='glorot_uniform', dtype=tf.float32),
            layers.Conv2D(1, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform', dtype=tf.float32)
        ])

    def build(self, input_shape):
        super().build(input_shape)
        input_channels = input_shape[-1]
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        self.sobel_x = tf.tile(sobel_x, [1, 1, input_channels, 1])
        self.sobel_y = tf.tile(sobel_y, [1, 1, input_channels, 1])
        self.alpha_subnetwork.build(input_shape)

    def compute_fourier_laplacian(self, x):
        shape = tf.shape(x)
        batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        static_shape = x.get_shape().as_list()
        if static_shape[1] is not None and static_shape[2] is not None:
            h_size = static_shape[1]
            w_size = static_shape[2]
        else:
            h_size = height
            w_size = width
        x_reshaped = tf.transpose(x, [0, 3, 1, 2])
        x_reshaped = tf.reshape(x_reshaped, [batch * channels, height, width])
        x_reshaped = x_reshaped + 1e-8
        x_complex = tf.cast(x_reshaped, tf.complex64)
        fft_x = tf.signal.fft2d(x_complex)
        k_y = tf.cast(tf.range(-h_size // 2, h_size // 2), dtype=tf.float32) / tf.cast(h_size, tf.float32)
        k_x = tf.cast(tf.range(-w_size // 2, w_size // 2), dtype=tf.float32) / tf.cast(w_size, tf.float32)
        k_x, k_y = tf.meshgrid(k_x, k_y)
        k_squared = -(k_x**2 + k_y**2) * 4.0 * np.pi**2
        k_squared = tf.signal.ifftshift(k_squared)
        k_squared = tf.expand_dims(k_squared, axis=0)
        fft_laplacian = fft_x * tf.cast(k_squared, tf.complex64)
        laplacian_complex = tf.signal.ifft2d(fft_laplacian)
        laplacian_real = tf.math.real(laplacian_complex)
        laplacian = tf.clip_by_value(laplacian_real, -10.0, 10.0)
        laplacian = tf.reshape(laplacian, [batch, channels, height, width])
        laplacian = tf.transpose(laplacian, [0, 2, 3, 1])
        return laplacian

    def call(self, inputs, training=None):
        try:
            x = tf.cast(inputs, tf.float32)
            x = tf.clip_by_value(x, -10.0, 10.0)
            laplacian = self.compute_fourier_laplacian(x)
            alpha = self.alpha_subnetwork(x, training=training)
            dt = 0.1 * alpha * laplacian
            updated_temperature = x + dt
            physics_loss = tf.constant(0.0, dtype=tf.float32)
            if training and self.use_physics_loss:
                physics_loss = self.compute_physics_loss(dt, updated_temperature)
            return updated_temperature, dt, physics_loss
        except Exception as e:
            logging.error(f"Error in FourierThermalLayer.call: {str(e)}")
            raise

    def compute_physics_loss(self, dt, updated_temperature):
        physics_loss = self.lambda_physics * 0.05 * tf.reduce_mean(tf.square(dt))
        grad_x = tf.nn.conv2d(updated_temperature, self.sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        grad_y = tf.nn.conv2d(updated_temperature, self.sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        grad_magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y) + 1e-8)
        mean_grad = tf.reduce_mean(grad_magnitude)
        var_grad = tf.math.reduce_variance(grad_magnitude)
        loss_grad = tf.square(mean_grad) + tf.square(var_grad - self.target_var)
        physics_loss += 0.5 * loss_grad
        return physics_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'lambda_physics': self.lambda_physics,
            'use_physics_loss': self.use_physics_loss,
            'target_var': self.target_var
        })
        return config

class StateSpaceModel(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.project_in = None
        self.project_out = None

    def build(self, input_shape):
        super().build(input_shape)
        input_channels = input_shape[-1]
        if input_channels != self.d_model:
            self.project_in = layers.Dense(self.d_model, name='project_in', dtype=tf.float32)
            self.project_out = layers.Dense(input_channels, name='project_out', dtype=tf.float32)
        self.W = self.add_weight(name='W', shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True, dtype=tf.float32)
        self.U = self.add_weight(name='U', shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True, dtype=tf.float32)
        self.V = self.add_weight(name='V', shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True, dtype=tf.float32)

    def call(self, inputs, training=None):
        try:
            x = tf.cast(inputs, tf.float32)
            x = tf.clip_by_value(x, -10.0, 10.0)
            original_shape = tf.shape(x)
            if self.project_in is not None:
                x = self.project_in(x)
            x_reshaped = tf.reshape(x, [-1, self.d_model])
            h = tf.nn.tanh(tf.linalg.matmul(x_reshaped, self.W))
            u = tf.nn.sigmoid(tf.linalg.matmul(x_reshaped, self.U))
            v = tf.nn.sigmoid(tf.linalg.matmul(x_reshaped, self.V))
            y = u * h + v * x_reshaped
            y = tf.reshape(y, tf.concat([original_shape[:-1], [self.d_model]], axis=0))
            if self.project_out is not None:
                y = self.project_out(y)
            y = tf.clip_by_value(y, -10.0, 10.0)
            return y
        except Exception as e:
            logging.error(f"Error in {self.__class__.__name__}.call: {str(e)}")
            raise

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config

class VisionMambaBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, use_physics_loss=True, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.use_physics_loss = use_physics_loss
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.fourier_layer = FourierThermalLayer(d_model=d_model, lambda_physics=5.0, use_physics_loss=True, target_var=0.1)
        self.ssm = StateSpaceModel(d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(4 * d_model, activation='relu', dtype=tf.float32),
            layers.Dense(d_model, dtype=tf.float32)
        ])
        self.physics_loss = None

    def build(self, input_shape):
        super().build(input_shape)
        self.norm1.build(input_shape)
        self.fourier_layer.build(input_shape)
        self.ssm.build(input_shape)
        self.norm2.build(input_shape)
        self.ffn.build(input_shape)

    def call(self, inputs, training=None):
        try:
            x = tf.cast(inputs, tf.float32)
            x_norm = self.norm1(x, training=training)
            x_fourier, state, physics_loss = self.fourier_layer(x_norm, training=training)
            if training and self.use_physics_loss:
                self.physics_loss = physics_loss
            y = self.ssm(x_fourier, training=training)
            y = self.norm2(y, training=training)
            y = self.ffn(y, training=training)
            return y
        except Exception as e:
            logging.error(f"Error in VisionMambaBlock.call: {str(e)}")
            raise

    def get_physics_loss(self):
        return self.physics_loss if self.physics_loss is not None else tf.constant(0.0, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'use_physics_loss': self.use_physics_loss})
        return config