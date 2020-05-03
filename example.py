
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from evaluations.robust_evaluate import evaluate
from attacks.fast_gradient_sign_method import FastGradientSignMethod
from attacks.basic_iterative_method import BasicIterativeMethod
from attacks.projected_gradient_descent import ProjectedGradientDescent

import preload.dataloader
import preload.datasets

from defenses.adversarial_train import adv_train

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def model_training(args, model, attack=None):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = preload.dataloader.DataLoader(
        preload.datasets.MNISTDataset('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size)
    test_loader = preload.dataloader.DataLoader(
        preload.datasets.MNISTDataset('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if attack is not None:
            adv_train(model, attack, device, train_loader, optimizer, epoch)
        else:
            train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        if attack is None:
            torch.save(model.state_dict(), "mnist_cnn.pt")
        else:
            torch.save(model.state_dict(), "mnist_cnn_{}.pt".format(attack.name))

def options():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Read Pre-trained Model')

    parser.add_argument('--eps', type=float, default=0.67, metavar='epsilon',
                        help='Max perturbation value (default: 0.67)')
    
    parser.add_argument('--alpha', type=float, default=0.033, metavar='alpha',
                        help='Max perturbation for each step (default: 0.033)')

    parser.add_argument('--iter-max', type=int, default=30, 
                        help='Max iteration for iterative attacks (default: 30)')

    args = parser.parse_args()

    return args

def evaluation(args, model, device, test_loader):
    print("Evaluating FGSM on MNIST:")
    fgsm = FastGradientSignMethod(lf=F.nll_loss, eps=args.eps)
    ori_acc, adv_acc = evaluate(model, fgsm, test_loader, device)
    print("Accuracy on original examples: {:.2f}%".format(100.*ori_acc))
    print("Accuracy on adversarial examples: {:.2f}%".format(100.*adv_acc))

    print("Evaluating BIM on MNIST:")
    bim = BasicIterativeMethod(lf=F.nll_loss, eps=args.eps, alpha=args.alpha, iter_max=args.iter_max)
    ori_acc, adv_acc = evaluate(model, bim, test_loader, device)
    print("Accuracy on original examples: {:.2f}%".format(100.*ori_acc))
    print("Accuracy on adversarial examples: {:.2f}%".format(100.*adv_acc))

    print("Evaluating PGD on MNIST:")
    pgd = ProjectedGradientDescent(lf=F.nll_loss, eps=args.eps, alpha=args.alpha, iter_max=args.iter_max)
    ori_acc, adv_acc = evaluate(model, pgd, test_loader, device)
    print("Accuracy on original examples: {:.2f}%".format(100.*ori_acc))
    print("Accuracy on adversarial examples: {:.2f}%".format(100.*adv_acc))

def main():
    args = options()

    use_cuda = not args.no_cuda and torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")


    test_loader = preload.dataloader.DataLoader(
        preload.datasets.MNISTDataset('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size)

    model = Net().to(device)
    start_point = model.state_dict()

    print("Normal training:")
    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn.pt"))
    else:
        model.load_state_dict(start_point)
        model_training(args, model)
    evaluation(args, model, device, test_loader)


    print("Adversarial training (FGSM):")
    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn_fgsm.pt"))
    else:
        model.load_state_dict(start_point)
        fgsm = FastGradientSignMethod(lf=F.nll_loss, eps=args.eps)
        model_training(args, model, fgsm)
    evaluation(args, model, device, test_loader)


    print("Adversarial training (BIM):")
    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn_bim.pt"))
    else:
        model.load_state_dict(start_point)
        bim = BasicIterativeMethod(lf=F.nll_loss, eps=args.eps, alpha=args.alpha, iter_max=args.iter_max)
        model_training(args, model, bim)
    evaluation(args, model, device, test_loader)

    
    
    print("Adversarial training (PGD):")
    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn_pgd.pt"))
    else:
        model.load_state_dict(start_point)
        pgd = ProjectedGradientDescent(lf=F.nll_loss, eps=args.eps, alpha=args.alpha, iter_max=args.iter_max)
        model_training(args, model, pgd)
    evaluation(args, model, device, test_loader)

if __name__ == "__main__":
    main()
