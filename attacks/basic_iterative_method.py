
from adversarial-examples-pytorch.attacks.attack import Attack


class BasicIterativeMethod(Attack):
    def __init__(self, lf, eps=0.25, alpha=0.01, iter_max=30, clip_min=-1.0, clip_max=1.0):
        super(FastGradientSignMethod, self).__init__()
        self.attack_parameters = { "epsilon" : eps, 
                                    "alpha" : alpha, 
                                    "iteration_max" : iter_max, 
                                    "clip_min" : clip_min, 
                                    "clip_max" : clip_max, 
                                    "loss_function" : lf 
                                    }

    def generate(self, model, x, labels):
        return BIM(model, x, labels, self.attack_parameters)

def BIM(model, x, labels, attack_parameters, **kwargs):
    lf = attack_parameters["loss_function"]
    epsilon = attack_parameters["epsilon"]
    alpha = attack_parameters["alpha"]
    iter_max = attack_parameters["iteration_max"]
    clip_min = attack_parameters["clip_min"]
    clip_max = attack_parameters["clip_max"]


    x_adv = x.clone().detach().requires_grad_(True).to(device)
    
    confidences = model(x_adv)
    pred = confidences.argmax(dim=1, keepdim=True)
    iteration = 0

    # Future work: Stop iteration when pred != labels
    while iteration != iter_max:
        model.zero_grad()
        loss = lf(confidences, labels)
        loss.backward()
        grad_sign = x_avd.grad.sign()
        x_adv += alpha * grad_sign

        x_adv = torch.clamp(x_adv, max(clip_min, x - epsilon), min(clip_max, x + epsilon))

        iteration += 1

    return x_adv