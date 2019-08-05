from model import Net
from PIL import Image
import torch
from torchvision import transforms
from my_dataset import MyDataset
from model import train, test
import argparse
import torch.optim as optim

fc1 = nn.Linear(28*28, 500)
fc2 = nn.Linear(500, 200)
fc3 = nn.Linear(200, 100)
fc4 = nn.Linear(100, 50)
fc5 = nn.Linear(50, 10)



class AttackNet(Net):
    pass

def classifies(model, image):
    outputs = model(image)
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    return predicted

def attack(model, image):
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    attacklabel = predicted
    while attacklabel == predicted:
        

def main():
    
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
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = AttackNet().to(device)
    model.load_state_dict(torch.load("mnist_cnn_20.pt"))
    model.eval()
    image = Image.open("attack.png").convert("L")
    #image.show()
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
    image = trans(image)
    print(classifies(model, image))

    '''
    train_loader = torch.utils.data.DataLoader(
        MyDataset("train", transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        MyDataset("test", transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn_20.pt")
    '''

if __name__ == "__main__":
    main()