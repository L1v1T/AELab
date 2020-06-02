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


def adv_guide_train(model, device, train_loader, guide_sets, optimizer, epoch, beta, epsilon):
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
    
    loss_sum = 0.0
    train_loss_sum = 0.0
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
        adv_pertur = - epsilon * torch.autograd.grad(L1, data_copy, create_graph=True)[0]
        min = torch.min(adv_pertur)
        max = torch.max(adv_pertur)
        mid = (max + min) / 2
        zero_mean = (max - min) / 2
        adv_pertur_norm = 2 * (adv_pertur - mid) / zero_mean
        # adv_data = data.clone().detach() + adv_pertur.clone().detach()
        # adv_output = model(adv_data)
        # adv_pred = adv_output.argmax(dim=1, keepdim=True)
        # guide_data, _ = guide_sample(guide_sets, adv_pred)
        # guide_data = guide_data.to(device)
        
        guide_data, _ = guide_sample(guide_sets, target_label)
        guide_data = guide_data.to(device)

        train_loss = F.nll_loss(output, target)
        guided_loss = F.mse_loss(adv_pertur_norm, guide_data - data)
        # guided_loss = F.kl_div(adv_pertur_norm, guide_data - data)
        loss = (1-beta)*train_loss + beta*guided_loss
        # loss = F.nll_loss(output, target) + F.mse_loss(guide_data - data, adv_pertur_norm)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        train_loss_sum += train_loss.item()
        guided_loss_sum += guided_loss.item()


    loss_sum /= len(train_loader)
    train_loss_sum /= len(train_loader)
    guided_loss_sum /= len(train_loader)

    print('Train Epoch: {} \tLoss: {:.6f}, Training Loss: {:.6f}, Guided Loss: {:.6f}'.format(
            epoch, loss_sum, train_loss_sum, guided_loss_sum))