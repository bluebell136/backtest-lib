#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np
import tensorflow as tf # install tf 2.0 or later

class BaseModel(abc.ABC, tf.keras.Model):

    def __init__(self, name, *args, **kwargs):
        """
        Neural network base class. Subclass BaseModel to define how (and what)
        a concrete Model should learn based on input data.

        :param name:
            str, model name
        """

        self._name = name # set private attribute in tf.keras.Model

    @abc.abstractmethod
    def build(self, input_shape):
        """
        Set as instance attributes all layers to be used in call method.

        :param input_shape:
            tuple, input shape where first dimension is batch_size
        """

        raise NotImplementedError("To be implemented in subclass.")

    @abc.abstractmethod
    def call(self, x):
        """
        Implement a single forward pass using the defined layers.

        :param x:
            tf.Tensor, batch
        """

        raise NotImplementedError("To be implemented in subclass.")
