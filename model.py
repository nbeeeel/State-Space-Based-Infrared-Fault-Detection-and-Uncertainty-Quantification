import tensorflow as tf
from tensorflow.keras import layers
from custom_layers import Float32Layer, VisionMambaBlock
import logging

def conv_block(x, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same", dtype=tf.float32)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(num_filters, 3, padding="same", dtype=tf.float32)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", dtype=tf.float32)(input)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, d_model=256, d_model1=512, num_classes=2, use_physics_loss=True):
    inputs = tf.keras.Input(input_shape, dtype=tf.float32)
    x = Float32Layer()(inputs)
    s1, p1 = encoder_block(x, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    vmb1 = VisionMambaBlock(d_model, use_physics_loss=use_physics_loss, name='vmb1')
    b1 = vmb1(p3)
    s4, p4 = encoder_block(b1, 512)
    vmb2 = VisionMambaBlock(d_model1, use_physics_loss=use_physics_loss, name='vmb2')
    b2 = vmb2(p4)
    d1 = decoder_block(b2, s4, 512)
    vmb3 = VisionMambaBlock(d_model1, use_physics_loss=use_physics_loss, name='vmb3')
    b3 = vmb3(d1)
    d2 = decoder_block(b3, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    x = layers.Dropout(0.2)(d4, training=True)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', dtype=tf.float32)(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='unet')
    model._vision_mamba_blocks = [vmb1, vmb2, vmb3]
    return model

class PhysicsAwareModel(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.physics_layers = getattr(base_model, '_vision_mamba_blocks', [])
        self._built = False

    def build(self, input_shape):
        super().build(input_shape)
        if not self._built:
            self.base_model.build(input_shape)
            self._built = True

    def initialize_physics_layers(self):
        if not self.physics_layers:
            logging.warning("No physics layers found in base model")
        else:
            logging.info(f"Initialized {len(self.physics_layers)} physics layers")

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def get_config(self):
        return super().get_config()