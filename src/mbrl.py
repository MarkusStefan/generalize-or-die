





class MBRLAlgorithm():
    '''
    Base class for Model-Based Reinforcement Learning algorithms.
    '''
    def __init__(self):
        pass



class Dyna(MBRLAlgorithm):
    '''
    Integrates model learning, planning, and policy learning.
    Uses the learned model to generate synthetic experiences to augment real experiences.
    '''
    def __init__(self):
        super().__init__()
        pass



class MBPO(MBRLAlgorithm):
    '''
    Trains an ensemble of neural network models for dynamics, 
    improves robustness by performing short model rollouts, 
    and uses model-free policy optimization (e.g., SAC or TD3) 
    based on both real and simulated data.
    '''
    def __init__(self):
        super().__init__()
        pass



class Dreamer(MBRLAlgorithm):
    '''
    Uses a world model to learn a compact representation of the environment, 
    enabling planning and policy learning in the latent space.
    -- Probably overkill due to resource requirements (GPU, TPU with >16GB RAM) --
    '''
    def __init__(self):
        super().__init__()
        pass