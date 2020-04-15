from adversarial-examples-pytorch.attacks.attack import Attack

class FastGradientSignMethod(Attack):
    def __init__(self, lf, eps=0.25, clip_min=-1.0, clip_max=1.0):
        super(FastGradientSignMethod, self).__init__()
        self.attack_parameters = { "epsilon" : eps, 
                                    "clip_min" : clip_min, 
                                    "clip_max" : clip_max, 
                                    "loss_function" : lf 
                                    }

    def generate(self, model, x, labels):
        return FGSM(model, x, labels, self.attack_parameters)

def FGMS(model, x, labels, attack_parameters):
    lf = attack_parameters["loss_function"]
    epsilon = attack_parameters["epsilon"]
    clip_min = attack_parameters["clip_min"]
    clip_max = attack_parameters["clip_max"]


    x_copy = x.clone().detach().to(device)
    x_adv = x_copy.clone().detach().requires_grad_(True).to(device)
    
    confidence = model(x_adv)
    model.zero_grad()
    loss = lf(confidence, labels)
    loss.backward()
    grad_sign = x_avd.grad.sign()
    x_copy += epsilon * grad_sign

    x_copy = torch.clamp(x_copy, clip_min, clip_max)

    return x_copy