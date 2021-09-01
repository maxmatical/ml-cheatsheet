from typing import Optional
from collections import defaultdict
import torch
import fastai 
from fastcore.basics import store_attr
from fastai.callback.core import Callback


class SAM(Callback):
    "Sharpness-Aware Minimization"
    def __init__(self, zero_grad=True, rho=0.05, eps=1e-12, **kwargs): 
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.state = defaultdict(dict)
        store_attr()

    def params(self): return self.learn.opt.all_params(with_grad=True)
    def _grad_norm(self): return torch.norm(torch.stack([p.grad.norm(p=2) for p,*_ in self.params()]), p=2)
    
    @torch.no_grad()
    def first_step(self):
        scale = self.rho / (self._grad_norm() + self.eps)
        for p,*_ in self.params():
            self.state[p]["e_w"] = e_w = p.grad * scale
            p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if self.zero_grad: self.learn.opt.zero_grad()
        
    @torch.no_grad()    
    def second_step(self):
        for p,*_ in self.params(): p.sub_(self.state[p]["e_w"])

    def before_step(self, **kwargs):
        self.first_step()
        self.learn.pred = self.model(*self.xb); self.learn('after_pred')
        self.loss_func(self.learn.pred, *self.yb).backward()
        self.second_step()

class BatchLossFilter(Callback):
    """ Callback that selects the hardest samples in every batch representing a percentage of the total loss"""

    def __init__(self, loss_perc=1., schedule_func:Optional[callable]=None):
        store_attr()

    def before_fit(self):
        self.run = not hasattr(self, "gather_preds")
        if not(self.run): return
        self.crit = self.learn.loss_func
        if hasattr(self.crit, 'reduction'): self.red = self.crit.reduction

    def before_batch(self):
        if not self.training or self.loss_perc == 1.: return
        with torch.no_grad():
            if hasattr(self.crit, 'reduction'):  setattr(self.crit, 'reduction', 'none')
            self.losses = self.crit(self.learn.model(self.x), self.y)
            if hasattr(self.crit, 'reduction'):  setattr(self.crit, 'reduction', self.red)
            self.losses /= self.losses.sum()
            idxs = torch.argsort(self.losses, descending=True)
            if self.schedule_func is not None: loss_perc = self.loss_perc * self.schedule_func(self.pct_train)
            else: loss_perc = self.loss_perc
            cut_idx = torch.argmax((self.losses[idxs].cumsum(0) > loss_perc).float())
            idxs = idxs[:cut_idx]
            self.learn.xb = tuple(xbi[idxs] for xbi in self.learn.xb)
            self.learn.yb = tuple(ybi[idxs] for ybi in self.learn.yb)

    def after_fit(self):
        if hasattr(self.learn.loss_func, 'reduction'):  setattr(self.learn.loss_func, 'reduction', self.red)
