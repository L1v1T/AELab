import torch



def evaluate(model, attack_method, dataloader, device):
    def test(model, device, test_loader, attack_method=None):
        model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if attack_method is not None:
                data = attack_method.generate(model, data, target)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        return correct, correct / len(test_loader.dataset)

    _, adv_acc = test(model, device, dataloader, attack_method)
    _, ori_acc = test(model, device, dataloader)
    return ori_acc, adv_acc