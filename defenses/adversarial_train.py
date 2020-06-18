import attacks.attack

import torch
import torch.nn.functional as F

import random

def adv_train(model, attack, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        adv_output = model(attack.generate(model, data, target))
        loss = attack.lf(output, target) + attack.lf(adv_output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss))


def adv_guide_train(model, device, train_loader, guide_sets, optimizer, epoch, 
                    beta, epsilon, weight_decay, gradient_decay):
    model.train()

    def guide_sample(datasets, adv_pred):
        def sample(dataset):
            import random
            idex = random.randint(0, len(dataset) - 1)
            return dataset[idex]
        
        data, target = sample(datasets[adv_pred[0]])
        databatch = data.unsqueeze(dim=0)
        labels = []
        labels.append(target)
        for i in range(1, len(adv_pred)):
            data, target = sample(datasets[adv_pred[i]])
            databatch = torch.cat([databatch, data.unsqueeze(dim=0)], dim=0)
            labels.append(target)

        labels = torch.tensor(labels)

        return databatch, labels
    
    # def l2_regular_loss(model, device):
    #     loss = None
    #     n = 0
    #     for name, param in model.named_parameters():
    #         if 'weight' in name:
    #             if loss is None:
    #                 loss = F.mse_loss(param, 
    #                                 torch.zeros(param.size()).to(device), 
    #                                 reduction='sum')
    #             else:
    #                 loss = loss + F.mse_loss(param, 
    #                                         torch.zeros(param.size()).to(device), 
    #                                         reduction='sum')
    #             n += param.numel()

    #     return loss / (2 * n)

    class LayerActivations:
        features = None
    
        def __init__(self, model):
            self.hook = model.conv2.register_forward_hook(self.hook_fn)

        def hook_fn(self, module, input, output):
            self.features = output
    
        def remove(self):
            self.hook.remove()
    
    # penultimate_layer = LayerActivations(model)

    loss_sum = 0.0
    train_loss_sum = 0.0
    # regular_loss_sum = 0.0
    # adv_regular_loss_sum = 0.0
    guided_loss_sum = 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # adv_data = attack.generate(model, data, target)

        # target selecting
        # randomly select a target label and calculate the adversarial perturbation
        target_label = target.clone().detach()
        for i in range(len(target_label)):
            while target_label[i] == target[i]:
                target_label[i] = random.randint(0, 9)
        data_copy = data.clone().detach().requires_grad_(True)
        output_copy = model(data_copy)
        L1 = F.nll_loss(output_copy, target_label)
        # print(torch.autograd.grad(L1, data_copy, create_graph=True)[0])
        adv_pertur = - torch.autograd.grad(L1, data_copy, create_graph=True)[0]
        min = torch.min(adv_pertur)
        max = torch.max(adv_pertur)
        mid = (max + min) / 2
        zero_mean = (max - min) / 2
        adv_pertur_norm = epsilon * (adv_pertur - mid) / zero_mean
        adv_data = data.clone().detach() + adv_pertur_norm

        # adv_pertur_norm = - epsilon * torch.tanh(torch.autograd.grad(L1, data_copy, create_graph=True)[0])

        # adv_data = data.clone().detach() + adv_pertur.clone().detach()
        # adv_output = model(adv_data)
        # adv_pred = adv_output.argmax(dim=1, keepdim=True)
        # guide_data, _ = guide_sample(guide_sets, adv_pred)
        # guide_data = guide_data.to(device)
        
        guide_data, _ = guide_sample(guide_sets, target_label)
        guide_data = guide_data.to(device)

        # model(adv_data)
        # adv_features = penultimate_layer.features
        # model(guide_data)
        # guide_features = penultimate_layer.features

        train_loss = F.nll_loss(output, target)
        # guided_loss = F.mse_loss(adv_pertur_norm, guide_data - data)
        guided_loss = F.mse_loss(adv_data, guide_data)
        # guided_loss = F.mse_loss(adv_features, guide_features)
        # regular_loss = l2_regular_loss(model, device)
        # adv_regular_loss = F.mse_loss(adv_pertur, torch.zeros(adv_pertur.size()).to(device))
        # guided_loss = F.kl_div(adv_pertur_norm, guide_data - data)
        # loss = (1-beta)*train_loss + beta*guided_loss + weight_decay*regular_loss + gradient_decay*adv_regular_loss
        loss = (1-beta)*train_loss + beta*guided_loss
        # loss = F.nll_loss(output, target) + F.mse_loss(guide_data - data, adv_pertur_norm)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        train_loss_sum += train_loss.item()
        # regular_loss_sum += regular_loss.item()
        # adv_regular_loss_sum += adv_regular_loss.item()
        guided_loss_sum += guided_loss.item()


    loss_sum /= len(train_loader)
    train_loss_sum /= len(train_loader)
    # regular_loss_sum /= len(train_loader)
    # adv_regular_loss_sum /= len(train_loader)
    guided_loss_sum /= len(train_loader)

    # penultimate_layer.remove()

    print('Train Epoch: {} \tLoss: {:.6f}, Training Loss: {:.6f}, Guided Loss: {:.6f}'.format(
            epoch, loss_sum, train_loss_sum, guided_loss_sum))

#     print('Train Epoch: {} \tLoss: {:.6f}, Training Loss: {:.6f}, \
# Gradient Regularization Loss: {:.3e}, L2 Regularization Loss: {:.3e}, \
# Guided Loss: {:.6f}'.format(
#             epoch, loss_sum, train_loss_sum, adv_regular_loss_sum, regular_loss_sum, guided_loss_sum))