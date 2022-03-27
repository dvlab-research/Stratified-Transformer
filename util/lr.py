from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR
import torch.optim as optim

class LambdaStepLR(LambdaLR):

  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v


class PolyLRwithWarmup(LambdaStepLR):
  """DeepLab learning rate policy"""

  def __init__(self, optimizer, max_iter, warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, power=1.0, last_step=-1):
    
    assert warmup == 'linear'
    def poly_with_warmup(s):
      coeff = (1 - s / (max_iter+1)) ** power
      if s <= warmup_iters:
        warmup_coeff = 1 - (1 - s / warmup_iters) * (1 - warmup_ratio)
      else:
        warmup_coeff = 1.0
      return coeff * warmup_coeff
    
    super(PolyLRwithWarmup, self).__init__(optimizer, poly_with_warmup, last_step)
    # torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
    # lr_lambda: A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.


class MultiStepWithWarmup(LambdaStepLR):
  def __init__(self, optimizer, milestones, gamma=0.1, warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, last_step=-1):

    assert warmup == 'linear'
    def multi_step_with_warmup(s):
      factor = 1.0
      for i in range(len(milestones)):
        if s < milestones[i]:
          break
        factor *= gamma
      
      if s <= warmup_iters:
        warmup_coeff = 1 - (1 - s / warmup_iters) * (1 - warmup_ratio)
      else:
        warmup_coeff = 1.0
      return warmup_coeff * factor

    super(MultiStepWithWarmup, self).__init__(optimizer, multi_step_with_warmup, last_step)


class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""

  def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)
    # torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
    # lr_lambda: A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.


class SquaredLR(LambdaStepLR):
  """ Used for SGD Lars"""

  def __init__(self, optimizer, max_iter, last_step=-1):
    super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**2, last_step)


class ExpLR(LambdaStepLR):

  def __init__(self, optimizer, step_size, gamma=0.9, last_step=-1):
    # (0.9 ** 21.854) = 0.1, (0.95 ** 44.8906) = 0.1
    # To get 0.1 every N using gamma 0.9, N * log(0.9)/log(0.1) = 0.04575749 N
    # To get 0.1 every N using gamma g, g ** N = 0.1 -> N * log(g) = log(0.1) -> g = np.exp(log(0.1) / N)
    super(ExpLR, self).__init__(optimizer, lambda s: gamma**(s / step_size), last_step)


def initialize_scheduler(optimizer, config, last_epoch=-1, scheduler_epoch=True, logger=None):
  # scheduler_epoch: the step_size are given in epoch num
  last_step = -1 if last_epoch < 0 else config.iter_per_epoch_train * (last_epoch + 1) - 1
  if scheduler_epoch:
    config.step_size = config.iter_per_epoch_train * config.step_size
    config.exp_step_size = config.iter_per_epoch_train * config.exp_step_size

  if config.scheduler == 'StepLR':
    return StepLR(optimizer, step_size=config.step_size, gamma=config.step_gamma, last_epoch=last_step)
  elif config.scheduler == 'PolyLR':
    return PolyLR(optimizer, max_iter=config.max_iter, power=config.poly_power, last_step=last_step)
  elif config.scheduler == 'PolyLRwithWarmup':
    return PolyLRwithWarmup(optimizer, max_iter=config.max_iter, warmup=config.warmup, warmup_iters=config.warmup_iters, warmup_ratio=config.warmup_ratio, power=config.poly_power, last_step=last_step)
  elif config.scheduler == 'SquaredLR':
    return SquaredLR(optimizer, max_iter=config.max_iter, last_step=last_step)
  elif config.scheduler == 'ExpLR':
    return ExpLR(optimizer, step_size=config.exp_step_size, gamma=config.exp_gamma, last_step=last_step)
  elif config.scheduler == 'OneCycleLR':
    return OneCycleLR(optimizer, max_lr=config.oc_max_lr, total_steps=config.max_iter, pct_start=config.oc_pct_start,
                      anneal_strategy=config.oc_anneal_strategy, div_factor=config.oc_div_factor,
                      final_div_factor=config.oc_final_div_factor, last_epoch=last_step)
  # (optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1)
  else:
    if logger is not None:
      logger.info('Scheduler not supported')
    else: print('Scheduler not supported')


if __name__ == '__main__':
  import torchvision.models as models
  model = models.vgg16()
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
  optimizer.param_groups[0]['initial_lr'] = 0.2 / 25.0
  optimizer.param_groups[0]['max_lr'] = 0.2
  optimizer.param_groups[0]['min_lr'] = 0.2 / 10000.0
  optimizer.param_groups[0]['max_momentum'] = 0.95
  optimizer.param_groups[0]['base_momentum'] = 0.85
  last_step = 2
  max_iter = 100
  # scheduler = PolyLR(optimizer, max_iter=max_iter, power=0.9, last_step=last_step)
  scheduler = OneCycleLR(optimizer, max_lr=0.2, total_steps=max_iter, pct_start=0.1, anneal_strategy='cos', div_factor=25.0,
             final_div_factor=10000.0, last_epoch=last_step)
  lr_list = []
  for epoch in range(max(last_step + 1, 0), min(max_iter, 100)):
    lrs = ', '.join(['{:.5e}'.format(x) for x in scheduler.get_last_lr()])
    print('epoch {} lrs {}'.format(epoch, lrs))
    lr_list.append(scheduler.get_last_lr()[0])
    scheduler.step()

  import numpy as np
  import matplotlib.pyplot as plt
  x = np.arange(max(last_step + 1, 0), min(max_iter, 100), 1)
  plt.title("function")
  plt.plot(x, lr_list)
  plt.show()