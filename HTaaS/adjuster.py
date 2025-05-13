import torch.optim as optim

class Adjuster:
    
    optimizer: optim.Optimizer
    
    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer

    def adjust_lr(self):
        pass
                
    def adjust_bs(self, bs: int) -> int:
        pass