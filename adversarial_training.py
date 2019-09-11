from model import load_model, train, test
from torchvision import datasets, transforms
from my_dataset import MyDataset
import torch.optim as optim
import torch
import argparse
import torch.nn.functional as F
import attack as A
from model import Net

def my_nll(model, data, target):
    output = model(data)
    return F.nll_loss(output, target)

def adv_loss(model, data, target, original_loss, attack_func, epsilon=0.33, alpha=0.5):

    adv_data = []
    adv_targets = []
    ori_data = data.clone().detach()
    # print(torch.equal(ori_data, data))
    for image in data:
        pass
        image = image.unsqueeze(0)
        adv_image, adv_target = attack_func(model, image, epsilon)
        adv_image = adv_image.squeeze(0)
        adv_data.append(adv_image.tolist())
        adv_targets.append(adv_target)
    adv_data = torch.tensor(adv_data)
    adv_targets = torch.tensor(adv_targets)
    # print(torch.equal(ori_data, data))
    # print(torch.equal(data, adv_data))

    # print(data)
    # print(target)
    # # adv_data, adv_target = attack_func(model, data, epsilon)
    # print(adv_data)
    # print(adv_targets)
    # exit(0)
    return alpha * original_loss(model, data, target) + \
        (1 - alpha) * original_loss(model, adv_data, target)
    # return original_loss(model, data, target)
    # output = model(data)
    # return F.nll_loss(output, target)

def adv_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(len(data))
        # print(len(data[0]))
        # print(len(data[0][0]))
        # print(len(data[0][0][0]))
        # print(data[0][0][0][0])
        # exit(0)
        # output = model(data)
        # loss = adv_loss(model, data, target, my_nll, A.fixed_I_FGMS, 7)
        loss = adv_loss(model, data, target, my_nll, A.FGMS, 0.7)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        MyDataset("train", transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        MyDataset("test", transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    # train_loader = torch.utils.data.DataLoader(
    #     MyDataset("train", transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     MyDataset("test", transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        adv_train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn_ad.pt")

if __name__ == "__main__":
    main()