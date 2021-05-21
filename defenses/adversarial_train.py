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
                    beta, epsilon, weight_decay, gradient_decay, attack='fgsm'):
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

    def aux_targets_gen(target):
        aux_target = torch.empty((9, 0), dtype=torch.int)
        for i in range(len(target)):
            target_list = list(range(10))
            target_list.remove(target[i])
            random.shuffle(target_list)
            target_tensor = torch.tensor(target_list).unsqueeze(dim=1)
            aux_target = torch.cat((aux_target, target_tensor), dim=-1)
        return aux_target

    loss_sum = 0.0
    train_loss_sum = 0.0
    # regular_loss_sum = 0.0
    # adv_regular_loss_sum = 0.0
    guided_loss_sum = 0.0

    def adv_gen(model, data, target_label, attack=attack, **kwargs):
        if attack == 'fgsm':
            epsilon = kwargs['epsilon']
            target_label = target_label.to(device)
            data_copy = data.clone().detach().requires_grad_(True)
            output_copy = model(data_copy)
            L1 = F.nll_loss(output_copy, target_label)
            adv_pertur = - torch.autograd.grad(L1, data_copy, create_graph=True)[0]
            # normalizing adversarial perturbation
            adv_max = torch.max(torch.max(adv_pertur, dim=-1)[0], dim=-1)[0]
            adv_min = torch.min(torch.min(adv_pertur, dim=-1)[0], dim=-1)[0]
            adv_mid = (adv_max + adv_min) / 2
            adv_zero = (adv_max - adv_min) / 2
            adv_norm = torch.ones_like(adv_pertur)
            for i in range(len(adv_pertur)):
                for j in range(len(adv_pertur[0])):
                    adv_norm[i][j] = (adv_pertur[i][j] - adv_mid[i][j].item()) / adv_zero[i][j].item()
            adv_norm = epsilon * adv_norm
            adv_data = data.clone().detach() + adv_norm
        if attack == 'pgd':
            epsilon = kwargs['epsilon']
            alpha = kwargs['alpha']
            iter_max = int(epsilon//alpha)
            target_label = target_label.to(device)
            adv_data = data.clone().detach().requires_grad_(True)

            for i in range(iter_max):
                output_copy = model(adv_data)
                L1 = F.nll_loss(output_copy, target_label)
                adv_pertur = - torch.autograd.grad(L1, adv_data, create_graph=True)[0]
                # normalizing adversarial perturbation
                adv_max = torch.max(torch.max(adv_pertur, dim=-1)[0], dim=-1)[0]
                adv_min = torch.min(torch.min(adv_pertur, dim=-1)[0], dim=-1)[0]
                adv_mid = (adv_max + adv_min) / 2
                adv_zero = (adv_max - adv_min) / 2
                adv_norm = torch.ones_like(adv_pertur)
                for i in range(len(adv_pertur)):
                    for j in range(len(adv_pertur[0])):
                        adv_norm[i][j] = (adv_pertur[i][j] - adv_mid[i][j].item()) / adv_zero[i][j].item()
                adv_norm = alpha * adv_norm
                adv_data = adv_data + adv_norm
        return adv_data
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # target selecting
        # generate adversarial perturbation for 9 target
        aux_targets = aux_targets_gen(target)
        guided_loss = 0.0
        for target_label in aux_targets:
            # generate adversarial example
            adv_data = adv_gen(model, data, target_label, attack=attack, epsilon=epsilon, alpha=0.033)

            guide_data, _ = guide_sample(guide_sets, target_label)
            guide_data = guide_data.to(device)
            
            guided_loss = F.mse_loss(adv_data, guide_data) + guided_loss
            break

        train_loss = F.nll_loss(output, target)
        loss = (1-beta)*train_loss + beta*guided_loss
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



def adv_guide_pgd_train(model, 
                        attack, 
                        device, 
                        train_loader, 
                        guide_sets, 
                        optimizer, 
                        epoch, 
                        beta, 
                        gamma, 
                        epsilon, 
                        weight_decay, 
                        gradient_decay):
    model.train()

    def aux_targets_gen(target):
        aux_target = torch.empty((9, 0), dtype=torch.int)
        for i in range(len(target)):
            target_list = list(range(10))
            target_list.remove(target[i])
            random.shuffle(target_list)
            target_tensor = torch.tensor(target_list).unsqueeze(dim=1)
            aux_target = torch.cat((aux_target, target_tensor), dim=-1)
        return aux_target

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
    adv_loss_sum = 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # target selecting
        # generate adversarial perturbation for 9 target
        # target_label = target.clone().detach()
        aux_targets = aux_targets_gen(target)
        guided_loss = 0
        adv_loss = 0
        for target_label in aux_targets:
            target_label = target_label.to(device)
            data_copy = data.clone().detach().requires_grad_(True)
            output_copy = model(data_copy)
            L1 = F.nll_loss(output_copy, target_label)
            adv_pertur = - torch.autograd.grad(L1, data_copy, create_graph=True)[0]
            
            # normalizing adversarial perturbation
            adv_max = torch.max(torch.max(adv_pertur, dim=-1)[0], dim=-1)[0]
            adv_min = torch.min(torch.min(adv_pertur, dim=-1)[0], dim=-1)[0]
            adv_mid = (adv_max + adv_min) / 2
            adv_zero = (adv_max - adv_min) / 2
            adv_norm = torch.ones_like(adv_pertur)
            for i in range(len(adv_pertur)):
                for j in range(len(adv_pertur[0])):
                    adv_norm[i][j] = (adv_pertur[i][j] - adv_mid[i][j].item()) / adv_zero[i][j].item()
            adv_norm = epsilon * adv_norm
            # min = torch.min(adv_pertur)
            # max = torch.max(adv_pertur)
            # mid = (max + min) / 2
            # zero_mean = (max - min) / 2
            # adv_pertur_norm = epsilon * (adv_pertur - mid) / zero_mean
            adv_data = data.clone().detach() + adv_norm

            guide_data, _ = guide_sample(guide_sets, target_label)
            guide_data = guide_data.to(device)

            # adversarial example output
            adv_output = model(attack.generate(model, data, target))
            guided_loss = F.mse_loss(adv_data, guide_data) + guided_loss
            adv_loss = attack.lf(adv_output, target) + adv_loss
        # for i in range(len(target_label)):
        #     while target_label[i] == target[i]:
        #         target_label[i] = random.randint(0, 9)
        

        train_loss = F.nll_loss(output, target)
        loss = (1-beta-gamma)*train_loss + beta*guided_loss + gamma*adv_loss
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        train_loss_sum += train_loss.item()
        guided_loss_sum += guided_loss.item()
        adv_loss_sum += adv_loss.item()


    loss_sum /= len(train_loader)
    train_loss_sum /= len(train_loader)
    guided_loss_sum /= len(train_loader)
    adv_loss_sum /= len(train_loader)


    print('Train Epoch: {} \tLoss: {:.6f}, Training Loss: {:.6f}, Guided Loss: {:.6f}, Adv Loss: {:.6f}'.format(
            epoch, loss_sum, train_loss_sum, guided_loss_sum, adv_loss_sum))