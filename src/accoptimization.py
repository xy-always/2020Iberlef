
"""
A MultistepAdamWeightDecayOptimizer can use larger batch_size in BERT 
which updates var after n steps 

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  accoptimizer = MultistepAdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      n=8,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    accoptimizer = tf.contrib.tpu.CrossShardOptimizer(accoptimizer)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
  # print(grads)

  train_op = accoptimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  print_op = tf.print(new_global_step,'----',learning_rate,'----',grads[-2][:5])
  with tf.control_dependencies([print_op]):
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  
  return train_op


class MultistepAdamWeightDecayOptimizer(optimizer.Optimizer):
  """A Multistep Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate=2e-5,
               weight_decay_rate=0.01,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               n=1, ##n steps per update
               exclude_from_weight_decay=None,
               name="MultistepAdamWeightDecayOptimizer"):
    """Constructs a MultistepAdamWeightDecayOptimizer."""
    super(MultistepAdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self._n = n  # Call Adam optimizer every n batches with accumulated grads
    self.exclude_from_weight_decay = exclude_from_weight_decay

    self._n_t = None  # n as tensor

  def _prepare(self):
    super(MultistepAdamWeightDecayOptimizer, self)._prepare()
    self._n_t = tf.convert_to_tensor(self._n, name="n")

  def _create_slots(self, var_list):
    """Create slot variables for MultistepAdamWeightDecayOptimizer with accumulated gradients.

    Like super class method, but additionally creates slots for the gradient
    accumulator `grad_acc` and the counter variable.
    """
    super(MultistepAdamWeightDecayOptimizer, self)._create_slots(var_list)
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=0 if self._n == 1 else 1,
                                   name="iter",
                                   colocate_with=first_var)
    for v in var_list:
      self._zeros_slot(v, "grad_acc", self._name)

  def _get_iter_variable(self):
    if tf.contrib.eager.in_eager_mode():
      graph = None
    else:
      graph = tf.get_default_graph()
    return self._get_non_slot_variable("iter", graph=graph)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    update_ops = []

    var_list = [v for g, v in grads_and_vars if g is not None]

    with ops.init_scope():
      self._create_slots(var_list)
    self._prepare()

    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue
      grad_acc = self.get_slot(param, "grad_acc")
      param_name = self._get_variable_name(param.name)
      m = tf.get_variable(name=param_name + "/adam_m",shape=param.shape.as_list(),
          dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer())
      v = tf.get_variable(name=param_name + "/adam_v",shape=param.shape.as_list(),
          dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer())
      
      ##apply adam for v
      def _apply_adam(grad_acc, grad, param, m, v):
        total_grad = (grad_acc + grad) / tf.cast(self._n_t, grad.dtype)
        # Standard Adam update.
        next_m = (
            tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, total_grad))
        next_v = (
            tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                      tf.square(total_grad)))
        update = next_m / (tf.sqrt(next_v) + self.epsilon)
        if self._do_use_weight_decay(param_name):
          update += self.weight_decay_rate * param
        update_with_lr = self.learning_rate * update
        next_param = param - update_with_lr
        adam_op = tf.group(param.assign(next_param), m.assign(next_m),
                        v.assign(next_v))
        with tf.control_dependencies([adam_op]):
          grad_acc_to_zero_op = grad_acc.assign(tf.zeros_like(grad_acc),
                                                use_locking=self._use_locking)
        return tf.group(adam_op, grad_acc_to_zero_op)
        
      ## accumulate gradients for var
      def _accumulate_gradient(grad_acc, grad):
        assign_op = tf.assign_add(grad_acc, grad, use_locking=self._use_locking)     
        return tf.group(assign_op) 
      ##apply adam or accumulate gradients for 'Var'
      update_op = tf.cond(tf.equal(self._get_iter_variable(), 0),
                   lambda: _apply_adam(grad_acc, grad, param, m, v),
                   lambda: _accumulate_gradient(grad_acc, grad))
      update_ops.append(update_op)

    ##do extra update ops for some var
    apply_updates = self._finish(update_ops, name_scope=name)      
    return apply_updates

  def _finish(self, update_ops, name_scope):
    """
    iter <- iter + 1 mod n
    """
    iter_ = self._get_iter_variable()
    with tf.control_dependencies(update_ops):
      with tf.colocate_with(iter_):
        update_iter = iter_.assign(tf.mod(iter_ + 1, self._n_t),
                                    use_locking=self._use_locking)
    return tf.group(
        *update_ops+[update_iter], name=name_scope)  

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
