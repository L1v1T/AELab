from attacks.attack import Attack
import torch

class FastGradientSignMethod(Attack):
    def __init__(self, lf, eps=0.25, clip_min=-1.0, clip_max=1.0, **kwargs):
        super(FastGradientSignMethod, self).__init__()
        self.attack_parameters = { "epsilon" : eps, 
                                    "clip_min" : clip_min, 
                                    "clip_max" : clip_max, 
                                    "loss_function" : lf 
                                    }

    def generate(self, model, x, labels, **kwargs):
        return FGSM(model, x, labels, self.attack_parameters, **kwargs)
        # return FGSM(model, x)

# def FGSM(model, ori_image, epsilon=0.25):
#     image = ori_image.clone().detach()
#     outputs = model(image)
#     _, predicted = torch.max(outputs, 1)

#     image_var = image.clone().detach().requires_grad_(True)
#     attackoutputs = model(image_var)
#     model.zero_grad()
#     loss = torch.nn.functional.nll_loss(attackoutputs, predicted)
#     loss.backward()
    
#     grad_sign = image_var.grad.sign()
#     image += epsilon * grad_sign

#     image = torch.clamp(image, 0, 1)
    
#     return image

def FGSM(model, x, labels, attack_parameters, **kwargs):
    lf = attack_parameters["loss_function"]
    epsilon = attack_parameters["epsilon"]
    clip_min = attack_parameters["clip_min"]
    clip_max = attack_parameters["clip_max"]

    device = kwargs["device"]
    x_copy = x.clone().detach().to(device)
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)
    model.zero_grad()
    confidence = model(x_adv)
    loss = lf(confidence, labels)
    loss.backward()
    grad_sign = x_adv.grad.sign()
    x_copy += epsilon * grad_sign

    x_copy = torch.clamp(x_copy, clip_min, clip_max)

    return x_copy