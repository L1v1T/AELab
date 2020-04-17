import torch


# class RobustEvaluate(object):
#     def __init__(self, **kwargs):
#         pass

#     def evaluate(self, model, attack_method, dataset, **kwargs):
#         def test(model, device, test_loader):
#             model.eval()
#             correct = 0
#             with torch.no_grad():
#                 for data, target in test_loader:
#                     data, target = data.to(device), target.to(device)
#                     output = model(data)
#                     pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#                     correct += pred.eq(target.view_as(pred)).sum().item()
#             return correct, correct / len(test_loader.dataset)



def evaluate(model, attack_method, dataloader, device):
    def test(model, device, test_loader, attack_method=None):
        model.eval()
        correct = 0
        for data, target in test_loader:
            if attack_method is not None:
                data = attack_method.generate(model, data, target, device=device)
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        return correct, correct / len(test_loader.dataset)

    _, adv_acc = test(model, device, dataloader, attack_method)
    _, ori_acc = test(model, device, dataloader)
    return ori_acc, adv_acc