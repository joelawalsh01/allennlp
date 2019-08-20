from typing import Any
import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class DropConnect(torch.nn.Module):
    """
    Implementation of DropConnect (Wan et al., 2013):

        http://proceedings.mlr.press/v28/wan13.html

    DropConnect applies dropout to weights in a neural network layer instead of the output.
    This module is intended to be used as a wrapper around other modules. This implementation
    is based on:

        https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py

    which is distributed under a BSD 3-Clause "New" or "Revised" License.

    Special shout out to @Stonesjtu for his work updating this code for PyTorch 1.0.

    Parameters
    ==========
    module : torch.nn.Module
        The target module.
    weights : List[str]
        List of target module weights to apply dropout to.
    dropout : float
        Dropout rate.
    """
    def __init__(self, module, weights, dropout=0, variational=False) -> None:
        super(DropConnect, self).__init__()
        self._module = module
        self._weights = weights
        self._dropout = dropout
        self._setup()

    @staticmethod
    def _never_flatten_parameters(*args, **kwargs) -> None:  # pylint: disable=unused-argument
        # Apparently there are issues using DropConnect on RNNs due to torch.nn.RNNBase's
        # flatten_parameters method. We get around this by replacing the method with this null
        # function which results in parameters never being flattened. We add this function as a
        # static method of the DropConnect module to avoid pickling issues.
        return

    def _setup(self):
        # As mentioned above, we need to prevent any subclass of torch.nn.RNNBase from compacting
        # weights. Since we want to allow DropConnect to be applied to any module, we need do this
        # for not only the target module, but also any of its submodules.
        for module in self._module.modules:
            if issubclass(type(module), torch.nn.RNNBase):
                module.flatten_parameters = self._never_flatten_parameters

        for weight in self.weights:
            # TODO: @loganiv. Should level be DEBUG or INFO?
            logger.debug('Applying dropconnect to weight: %s', weight)
            weight_tensor = getattr(self.module, weight)
            del self.module._parameters[weight]
            self.module.register_parameter(weight + '_raw', torch.nn.Parameter(weight_tensor.data))

    def _setweights(self) -> None:
        for weight in self.weights:
            raw_weight_tensor = getattr(self.module, weight + '_raw')
            weight_tensor = torch.nn.functional.dropout(raw_weight_tensor,
                                                        p=self.dropout,
                                                        training=self.training)
            setattr(self.module, weight, torch.nn.Parameter(weight_tensor))

    def forward(self, *args, **kwargs) -> Any:  # TODO: @rloganiv. Not sure what right return type is...
        self._setweights()
        return self._module.forward(*args, **kwargs)
