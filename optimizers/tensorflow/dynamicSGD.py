from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.training import gen_training_ops
from tensorflow.python.util.tf_export import keras_export


# import horovod.tensorflow as hvd

# T = 8 * hvd.size()

@keras_export("keras.optimizers.DynamicSGD")
class DynamicSGD(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.5,
                 nesterov=False,
                 name="DynamicSGD",
                 **kwargs):
        super(DynamicSGD, self).__init__(name, **kwargs)
        self.k = 1.0
        self.iter = 1
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        # compensation = hvd.size()
        # if (iters - iters0) < T:
        #     compensation = 1 + (iters - iters0) * (hvd.size() - 1) / T

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(DynamicSGD, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
            self._get_hyper("momentum", var_dtype))

    def set_with_compensation(self, iter_now, ratio):
        self.iter = iter_now
        self.k = ratio

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            print("apply dense momentum = ", momentum_var)
            print(coefficients["momentum"].op.name)
            return gen_training_ops.ResourceApplyKerasMomentum(
                var=var.handle,
                accum=momentum_var.handle,
                lr=coefficients["lr_t"],
                grad=grad,
                momentum=coefficients["momentum"],
                use_locking=self._use_locking,
                use_nesterov=self.nesterov)
        else:
            return gen_training_ops.ResourceApplyGradientDescent(
                var=var.handle,
                alpha=coefficients["lr_t"],
                delta=grad,
                use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                                 **kwargs):
        if self._momentum:
            return super(DynamicSGD, self)._resource_apply_sparse_duplicate_indices(
                grad, var, indices, **kwargs)
        else:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                            or self._fallback_apply_state(var_device, var_dtype))

            return gen_resource_variable_ops.ResourceScatterAdd(
                resource=var.handle,
                indices=indices,
                updates=-grad * coefficients["lr_t"])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        momentum_var = self.get_slot(var, "momentum")
        print("apply sparse momentum = ", momentum_var)
        return gen_training_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=momentum_var.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            indices=indices,
            momentum=coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)

    def get_config(self):
        config = super(DynamicSGD, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._initial_decay,
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
        })
        return config
