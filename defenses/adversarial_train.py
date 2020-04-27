import attacks.attack

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


def adv_guide_train(model, attack, device, train_loader, optimizer, epoch):
    pass