from attacks.attack import Attack
import torch
import torch.nn.functional as F

class FastGradientSignMethod(Attack):
    def __init__(self, lf=F.nll_loss, eps=0.67, clip_min=-1.0, clip_max=1.0, **kwargs):
        super(FastGradientSignMethod, self).__init__()
        self.lf = lf
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.name = "fgsm"

    def generate(self, model, x, labels, **kwargs):
        self.update_params(**kwargs)
        return FGSM(model, 
                    x, 
                    labels, 
                    target=self.target, 
                    lf=self.lf, 
                    eps=self.eps, 
                    clip_min=self.clip_min, 
                    clip_max=self.clip_max)

    def update_params(self, **kwargs):
        if 'target' in kwargs:
            self.target = kwargs['target']
        else:
            self.target = None
        if 'lf' in kwargs:
            self.lf = kwargs['lf']
        if 'eps' in kwargs:
            self.eps = kwargs['eps']
        if 'clip_min' in kwargs:
            self.clip_min = kwargs['clip_min']
        if 'clip_max' in kwargs:
            self.clip_max = kwargs['clip_max']

def FGSM(model, x, labels, target=None, lf=F.nll_loss, eps=0.67, clip_min=-1.0, clip_max=1.0):
    x_copy = x.clone().detach()
    x_adv = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    confidence = model(x_adv)
    if target is None:
        loss = lf(confidence, labels)
    else:
        loss = lf(confidence, target)
    loss.backward()
    grad_sign = x_adv.grad.sign()
    if target is None:
        x_copy += eps * grad_sign
    else:
        x_copy -= eps * grad_sign

    x_copy = torch.clamp(x_copy, clip_min, clip_max)

    return x_copy