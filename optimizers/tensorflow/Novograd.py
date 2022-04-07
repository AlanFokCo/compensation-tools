from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.train import MomentumOptimizer
import tensorflow as tf

class NovoGrad(MomentumOptimizer):

  def __init__(self,
               learning_rate=1.0,
               beta1=0.95,
               beta2=0.98,
               epsilon=1e-8,
               weight_decay=0.0,
               grad_averaging=False,
               use_locking=False,
               name='NovoGrad'):
    super(NovoGrad, self).__init__(learning_rate, momentum=beta1,
                                   use_locking=use_locking, name=name,
                                   use_nesterov=False)
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._wd  = weight_decay
    self._grad_averaging  = grad_averaging
    self._grads_ema = None

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    len_vars = len(grads_and_vars)
    if self._grads_ema is None:
      self._grads_ema = [None] * len_vars
      for i in range(len_vars):
        self._grads_ema[i] = tf.get_variable(name="nvgrad2_ema" + str(i),
                                     shape=[], dtype=tf.float32,
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=False)

    # compute ema for grads^2 for each layer
    for i, (grad, var) in enumerate(grads_and_vars):
      g_2 = tf.reduce_sum(tf.square(x=tf.cast(grad, tf.float32)))
      self._grads_ema[i] = tf.cond(tf.equal(self._grads_ema[i], 0.),
                  lambda: g_2,
                  lambda: self._grads_ema[i]*self._beta2 + g_2*(1.-self._beta2)
                  )

      grad *= 1.0 / tf.sqrt(self._grads_ema[i] + self._epsilon)
      # weight decay
      if (self._wd > 0.):
        grad += (self._wd * var)
      # Momentum --> SAG
      if self._grad_averaging:
        grad *= (1.-self._beta1)
      grads_and_vars[i] = (grad, var)

    # call Momentum to do update
    return super(NovoGrad, self).apply_gradients(
         grads_and_vars, global_step=global_step, name=name)