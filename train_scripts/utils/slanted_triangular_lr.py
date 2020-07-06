import torch

class STLR(torch.optim.lr_scheduler._LRScheduler):
    '''https://gist.github.com/ceshine/ff32968bafc6fead87d7b6233ad8ab69   
    Parameters:
        optimizer: torch.optim.Optimizer
        max_mul: peak learning rate multiplier. Increases LR up to lr*max_mul, then decreases again
        ratio: Ratio of LR decrease to increase ratio (1=equal, higher= increase shorter) 
        steps_per_cycle: Over how many steps to run the schedule, the full training duration in steps
        decay: Factor by which to decrease max_mul after each cycle (2nd scheduler cycle: `max_mul = max_mul * decay`)
        last_epoch: TODO unclear

    '''
    def __init__(self, optimizer, max_mul, ratio: float, steps_per_cycle: int, decay=1, last_epoch=-1):

        self.max_mul = max_mul - 1
        self.turning_point = steps_per_cycle // (ratio + 1)
        self.steps_per_cycle = steps_per_cycle
        self.decay = decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        residual = self.last_epoch % self.steps_per_cycle
        multiplier = self.decay ** (self.last_epoch // self.steps_per_cycle)
        if residual <= self.turning_point:
            multiplier *= self.max_mul * (residual / self.turning_point)
        else:
            multiplier *= self.max_mul * (
                (self.steps_per_cycle - residual) /
                (self.steps_per_cycle - self.turning_point))
        return [lr * (1 + multiplier) for lr in self.base_lrs]