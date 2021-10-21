import tensorflow as tf
import tensorflow.keras as keras

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=1000):
        super(LearningRateSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class GumbelKLSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, vqvae, kl_steps = 18189, gumbel_steps = 303150, freeze_steps = 10105):
        """
        learning_rate = 1e-4(fixed)
        vqvae => vqvae model
        kl_steps => from 0 to 6.6 until 10105(5epoch) + 18189 steps (kl_steps)
        gumbel_steps => from 1 to 0 until 10105(5epoch) + 303150 steps (gumbel_steps)
        freeze_steps => gumbel = 1, kl = 0
        """
        super(GumbelKLSchedule, self).__init__()
        self.learning_rate = learning_rate
        self.kl_steps = kl_steps
        self.gumbel_steps = gumbel_steps
        self.vqvae = vqvae
        self.freeze_steps = freeze_steps
        self.step = 0

    def __call__(self, step):
        self.step += 1
        if self.step > self.freeze_steps:
            step_forward = self.step - self.freeze_steps
            if step_forward <= self.kl_steps :
                kl_ratio = 6.6 / self.kl_steps * step_forward
                self.vqvae.set_kl_ratio(kl_ratio)
            if step_forward <= self.gumbel_steps :
                gumbel_tem = 1 - 1 / self.gumbel_steps * step_forward + 0.06
                self.vqvae.set_gumbel_temperature(gumbel_tem)

        return self.learning_rate

import numpy as np


class GumbelAndKLRatioCallback(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, kl_steps = 18189, gumbel_steps = 303150, freeze_steps = 10105):
        super(GumbelAndKLRatioCallback, self).__init__()
        self.kl_steps = kl_steps
        self.gumbel_steps = gumbel_steps
        self.freeze_steps = freeze_steps
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step > self.freeze_steps:
            step_forward = self.step - self.freeze_steps
            if step_forward <= self.kl_steps :
                kl_ratio = 6.6 / self.kl_steps
                self.model.kl_weight.assign_add(kl_ratio)
            if step_forward <= self.gumbel_steps :
                gumbel_tem = 1 / self.gumbel_steps
                self.model.temper_weight.assign_add(-gumbel_tem)
