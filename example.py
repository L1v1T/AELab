
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
from defenses.adversarial_train import adv_guide_train

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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class TrainMethod(object):
    def __init__(self, model, device, train_loader, optimizer, **kwargs):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.update_kwargs(**kwargs)

    def update_kwargs(self, **kwargs):
        error = "Sub-classes must implement 'update_kwargs' method."
        raise NotImplementedError(error)

    def train(self, epoch):
        normal_train(self.model, self.device, self.train_loader, self.optimizer, epoch)

def normal_train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss))


class NormalTrain(TrainMethod):
    def __init__(self, model, device, train_loader, optimizer, **kwargs):
        super(NormalTrain, self).__init__(model, device, train_loader, optimizer, **kwargs)

    def update_kwargs(self, **kwargs):
        pass

    def train(self, epoch):
        normal_train_show_l2(self.model, 
                            self.device, 
                            self.train_loader, 
                            self.optimizer, 
                            epoch)

def normal_train_show_l2(model, 
                        device, 
                        train_loader, 
                        optimizer, 
                        epoch):
    model.train()
    loss_sum = 0.0
    train_loss_sum = 0.0
    regular_loss_sum = 0.0

    def l2_regular_loss(model, device):
        loss = 0
        n = 0
        for paramkey in model.state_dict().keys():
            if 'bias' in paramkey:
                pass
            else:
                # loss += torch.norm(model.state_dict()[paramkey])
                loss += F.mse_loss(model.state_dict()[paramkey], 
                            torch.zeros(model.state_dict()[paramkey].size()).to(device), 
                            reduction='sum')
                n += model.state_dict()[paramkey].numel()
        return loss / (2 * n)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = F.nll_loss(output, target)
        regular_loss = l2_regular_loss(model, device)
        loss = train_loss
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        train_loss_sum += train_loss.item()
        regular_loss_sum += regular_loss.item()

    loss_sum /= len(train_loader)
    train_loss_sum /= len(train_loader)
    regular_loss_sum /= len(train_loader)

    print('Train Epoch: {} \tLoss: {:.6f}, Training Loss: {:.6f}, L2 Regularization Loss: {:.6f}'.format(
            epoch, loss_sum, train_loss_sum, regular_loss_sum))

class L2RegularTrain(TrainMethod):
    def __init__(self, model, device, train_loader, optimizer, **kwargs):
        super(L2RegularTrain, self).__init__(model, device, train_loader, optimizer, **kwargs)
    
    def update_kwargs(self, **kwargs):
        self.weight_decay = kwargs['weight_decay']

    def train(self, epoch):
        l2_regular_train(self.model, 
                        self.device, 
                        self.train_loader, 
                        self.optimizer, 
                        self.weight_decay,
                        epoch)

def l2_regular_train(model, device, train_loader, optimizer, weight_decay, epoch):
    model.train()
    loss_sum = 0.0
    train_loss_sum = 0.0
    regular_loss_sum = 0.0

    def l2_regular_loss(model, device):
        loss = 0
        n = 0
        for paramkey in model.state_dict().keys():
            if 'bias' in paramkey:
                pass
            else:
                # loss += torch.norm(model.state_dict()[paramkey])
                loss += F.mse_loss(model.state_dict()[paramkey], 
                            torch.zeros(model.state_dict()[paramkey].size()).to(device), 
                            reduction='sum')
                n += model.state_dict()[paramkey].numel()
        return loss / (2 * n)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = F.nll_loss(output, target)
        regular_loss = l2_regular_loss(model, device)
        loss = train_loss + weight_decay * regular_loss
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        train_loss_sum += train_loss.item()
        regular_loss_sum += regular_loss.item()

    loss_sum /= len(train_loader)
    train_loss_sum /= len(train_loader)
    regular_loss_sum /= len(train_loader)

    print('Train Epoch: {} \tLoss: {:.6f}, Training Loss: {:.6f}, L2 Regularization Loss: {:.6f}'.format(
            epoch, loss_sum, train_loss_sum, regular_loss_sum))

class AdversarialTrain(TrainMethod):
    def __init__(self, model, device, train_loader, optimizer, **kwargs):
        super(AdversarialTrain, self).__init__(model, device, train_loader, optimizer, **kwargs)

    def update_kwargs(self, **kwargs):
        self.attack = kwargs['attack']

    def train(self, epoch):
        adv_train(self.model, 
                self.attack, 
                self.device, 
                self.train_loader, 
                self.optimizer, 
                epoch)

class AdversarialGuidedTrain(TrainMethod):
    def __init__(self, model, device, train_loader, optimizer, **kwargs):
        super(AdversarialGuidedTrain, self).__init__(model, device, train_loader, optimizer, **kwargs)

    def update_kwargs(self, **kwargs):
        self.guide_sets = kwargs['guide_sets']
        self.epsilon = kwargs['epsilon']
        self.beta = kwargs['beta']

    def train(self, epoch):
        adv_guide_train(self.model, 
                        self.device, 
                        self.train_loader, 
                        self.guide_sets, 
                        self.optimizer, 
                        epoch, 
                        self.beta, 
                        self.epsilon)


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

    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def model_training(args, model, train_method, device, test_loader, scheduler):
    for epoch in range(1, args.epochs + 1):
        train_method.train(epoch)
        # if attack is not None:
        #     adv_train(model, attack, device, train_loader, optimizer, epoch)
        # else:
        #     train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # if args.save_model:
    #     if attack is None:
    #         torch.save(model.state_dict(), "mnist_cnn.pt")
    #     else:
    #         torch.save(model.state_dict(), "mnist_cnn_{}.pt".format(attack.name))

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

    parser.add_argument('--beta', type=float, default=0.9, metavar='beta',
                        help='Trade off factor of two loss function (default: 0.9)')

    parser.add_argument('--weight-decay', type=float, default=1.0, metavar='Weight decay', 
                        help='Weight decay factor of l2 regularization (default: 1.0)')

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

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_set = preload.datasets.MNISTDataset('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))

    train_loader = preload.dataloader.DataLoader(train_set, batch_size=args.batch_size)
    
    test_loader = preload.dataloader.DataLoader(
        preload.datasets.MNISTDataset('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size)
    

    model = Net().to(device)
    start_point = model.state_dict()
    print(start_point['fc2.bias'])

    print("\nNormal training:")
    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn.pt"))
    else:
        model.load_state_dict(start_point)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        normal_method = NormalTrain(model, device, train_loader, optimizer)
        model_training(args, model, normal_method, device, test_loader, scheduler)
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")
    evaluation(args, model, device, test_loader)

    print(start_point['fc2.bias'])

    print("\nNormal training with L2 regularization:")
    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn_l2_regular.pt"))
    else:
        model.load_state_dict(start_point)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        l2_method = L2RegularTrain(model, 
                                    device, 
                                    train_loader, 
                                    optimizer, 
                                    weight_decay=args.weight_decay)
        model_training(args, model, l2_method, device, test_loader, scheduler)
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")
    evaluation(args, model, device, test_loader)


    # print("\nAdversarial training (FGSM):")
    # if args.load_model:
    #     model.load_state_dict(torch.load("mnist_cnn_fgsm.pt"))
    # else:
    #     model.load_state_dict(start_point)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #     fgsm = FastGradientSignMethod(lf=F.nll_loss, eps=args.eps)
    #     adv_method = AdversarialTrain(model, device, train_loader, optimizer, attack=fgsm)
    #     model_training(args, model, adv_method, device, test_loader, scheduler)
    #     if args.save_model:
    #         torch.save(model.state_dict(), "mnist_cnn_{}.pt".format(fgsm.name))
    # evaluation(args, model, device, test_loader)


    # print("\nAdversarial training (BIM):")
    # if args.load_model:
    #     model.load_state_dict(torch.load("mnist_cnn_bim.pt"))
    # else:
    #     model.load_state_dict(start_point)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #     bim = BasicIterativeMethod(lf=F.nll_loss, eps=args.eps, alpha=args.alpha, iter_max=args.iter_max)
    #     adv_method = AdversarialTrain(model, device, train_loader, optimizer, attack=bim)
    #     model_training(args, model, adv_method, device, test_loader, scheduler)
    #     if args.save_model:
    #         torch.save(model.state_dict(), "mnist_cnn_{}.pt".format(bim.name))
    # evaluation(args, model, device, test_loader)

    
    
    # print("\nAdversarial training (PGD):")
    # if args.load_model:
    #     model.load_state_dict(torch.load("mnist_cnn_pgd.pt"))
    # else:
    #     model.load_state_dict(start_point)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #     pgd = ProjectedGradientDescent(lf=F.nll_loss, eps=args.eps, alpha=args.alpha, iter_max=args.iter_max)
    #     adv_method = AdversarialTrain(model, device, train_loader, optimizer, attack=pgd)
    #     model_training(args, model, adv_method, device, test_loader, scheduler)
    #     if args.save_model:
    #         torch.save(model.state_dict(), "mnist_cnn_{}.pt".format(pgd.name))
    # evaluation(args, model, device, test_loader)


    # print("\nAdversarial guided training:")
    # if args.load_model:
    #     model.load_state_dict(torch.load("mnist_cnn_adv_guided.pt"))
    # else:
    #     model.load_state_dict(start_point)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #     guide_sets = make_guide_set(train_set, size=1000)
    #     adv_guided_method = AdversarialGuidedTrain(model, 
    #                             device, 
    #                             train_loader, 
    #                             optimizer, 
    #                             guide_sets=guide_sets, 
    #                             epsilon=args.eps, 
    #                             beta=args.beta)
    #     model_training(args, model, adv_guided_method, device, test_loader, scheduler)
    #     if args.save_model:
    #         torch.save(model.state_dict(), "mnist_cnn_adv_guided.pt")
    # evaluation(args, model, device, test_loader)
        



def make_guide_set(dataset, size=1):
    guide_sets = []
    import random
    for i in range(10):
        subset_index = []
        count = 0
        while count < 1000:
            rand_index = random.randint(0, len(dataset) - 1)
            if dataset[rand_index][1] == i:
                subset_index.append(rand_index)
                count += 1
        guide_sets.append(torch.utils.data.Subset(dataset, subset_index))
    
    return guide_sets

# def adversarial_guide_training():
#     args = options()

#     use_cuda = not args.no_cuda and torch.cuda.is_available()


#     device = torch.device("cuda" if use_cuda else "cpu")


#     test_loader = preload.dataloader.DataLoader(
#         preload.datasets.MNISTDataset('../data', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.5,), (0.5,))
#                        ])),
#         batch_size=args.test_batch_size)

#     model = Net().to(device)
#     print("Adversarial guide training:")
#     if args.load_model:
#         model.load_state_dict(torch.load("mnist_cnn_ag.pt"))
#     else:
#         train_set = preload.datasets.MNISTDataset('../data', train=True, download=True,
#                                     transform=transforms.Compose([
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.5,), (0.5,))
#                                     ]))
#         guide_set = make_guide_set(train_set, size=1000)
#         adv_guide_train(model, device, train_loader, guide_sets, optimizer, epochs, epsilon)

if __name__ == "__main__":
    main()
