'''World Models for Model-Based Reinforcement Learning'''

from abc import ABC, abstractmethod

import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F

# import engression
from engression.models import StoNet


class WorldModel(ABC):
    '''
    Base class for World Models used in Model-Based Reinforcement Learning.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, state, action):
        '''
        Predict the next state given the current state and action.
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train the world model using collected data.
        '''
        pass



class MLPDynamicsModel(WorldModel):
    '''
    A simple Multi-Layer Perceptron (MLP) based dynamics model.
    '''
    def __init__(self):
        super().__init__()
        pass

    def predict(self, state, action):
        # Implement MLP prediction logic here
        pass

    def train(self, data):
        # Implement MLP training logic here
        pass


class EngressionDynamicsModel(WorldModel):
    '''
    Engression model (=energy regression) facilitate generalization.
    '''
    def __init__(self):
        super().__init__()
        pass

    def predict(self, state, action):
        pass

    def train(self, data):
        pass