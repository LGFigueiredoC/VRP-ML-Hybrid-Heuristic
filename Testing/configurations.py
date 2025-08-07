import torch
import os

class Configuration:
    def __init__(self, data_set, model_dir):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        torch.manual_seed(0) # Para reproduzir os resultados

        torch.set_default_tensor_type(torch.DoubleTensor)

        self.data_set = data_set
        self.model_dir = model_dir

        self.cpu_num = int((os.cpu_count()/2)-1)

        self.Q = 1
        self.alpha = 1
        self.beta = 1
        self.decay = 0.05
        self.initCity = 0
        self.seed = [5,7,11,13,17]
        self.probNew = 0.1