
from attacks.attack import Attack
import torch
import torch.nn.functional as F


class BasicIterativeMethod(Attack):
    def __init__(self, lf=F.nll_loss, eps=0.67, alpha=0.033, iter_max=30, clip_min=-1.0, clip_max=1.0, **kwargs):
        super(BasicIterativeMethod, self).__init__()
        self.lf = lf
        self.eps = eps
        self.alpha = alpha
        self.iter_max = iter_max
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, model, x, labels, **kwargs):
        self.update_params(**kwargs)
        return BIM(
                    model, 
                    x, 
                    labels, 
                    lf=self.lf, 
                    eps=self.eps, 
                    alpha=self.alpha, 
                    iter_max=self.iter_max, 
                    clip_min=self.clip_min, 
                    clip_max=self.clip_max)

    
    def update_params(self, **kwargs):
        if 'lf' in kwargs:
            self.lf = kwargs['lf']
        if 'eps' in kwargs:
            self.eps = kwargs['eps']
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'iter_max' in kwargs:
            self.iter_max = kwargs['iter_max']
        if 'clip_min' in kwargs:
            self.clip_min = kwargs['clip_min']
        if 'clip_max' in kwargs:
            self.clip_max = kwargs['clip_max']

def BIM(model, x, labels, 
        lf=F.nll_loss, eps=0.67, alpha=0.033, iter_max=30, clip_min=-1.0, clip_max=1.0, **kwargs):

    x_copy = x.clone().detach()
    
    iteration = 0

    clip_tensor_min = x.clone().detach() * 0 + clip_min
    clip_tensor_max = x.clone().detach() * 0 + clip_max

    # Future work: Stop iteration when pred != labels
    while iteration != iter_max:
        x_adv = x_copy.clone().detach().requires_grad_(True)
        model.zero_grad()
        confidences = model(x_adv)
        loss = lf(confidences, labels)
        loss.backward()
        grad_sign = x_adv.grad.sign()
        x_copy += alpha * grad_sign

        x_copy = torch.max(x_copy, torch.max(clip_tensor_min, x - eps))
        x_copy = torch.min(x_copy, torch.min(clip_tensor_max, x + eps))

        iteration += 1

    return x_copy