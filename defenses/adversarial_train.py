import attacks.attack

import torch.nn.functional as F

def adv_train(model, attack, device, train_loader, optimizer, epoch):
    model.train()
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        adv_output = model(attack.generate(model, data, target))
        loss = attack.lf(output, target) + attack.lf(adv_output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def adv_guide_train(model, device, train_loader, guide_sets, optimizer, epoch, epsilon):
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
    
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # adv_data = attack.generate(model, data, target)

        data_copy = data.clone().detach().requires_grad_(True)
        output_copy = model(data_copy)
        L1 = F.nll_loss(output_copy, target)
        adv_pertur = epsilon * torch.autograd.grad(L1, data_copy, create_graph=True).sign()

        adv_data = data.clone().detach() + adv_pertur.clone().detach()
        adv_output = model(adv_data)
        adv_pred = adv_output.argmax(dim=1, keepdim=True)
        guide_data, _ = guide_sample(guide_sets, adv_pred)
        guide_data = guide_data.to(device)
        
        loss = F.nll_loss(output, target) + F.mse_loss(data - guide_data, adv_pertur)
        loss.backward()
        optimizer.step()