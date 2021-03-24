import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

import preload.dataloader
import preload.datasets

from interpreters.grad import GRAD
from attacks.fast_gradient_sign_method import FastGradientSignMethod

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


def saliency(interpreter, x, y, root="val.jpeg"):
    saliency_map = interpreter(x, y)
    for i in range(len(saliency_map)):
        max, min = saliency_map[i].max(), saliency_map[i].min()
        saliency_map[i] = (saliency_map[i] - min) / (max - min)
    x = torch.cat((x, saliency_map))
    torchvision.utils.save_image(x, root)

def adv_saliency_evaluate(model, device, x, y, x_t, y_t, name):
    intprtr = GRAD(model).to(device)
    saliency(intprtr, x, y, root=name+"_normal.jpeg")
    fgsm = FastGradientSignMethod(lf=F.nll_loss, eps=0.67)
    x_adv = fgsm.generate(model, x, y_t)
    saliency(intprtr, x_adv, y_t, root=name+"_adv.jpeg")
    saliency(intprtr, x_t, y_t, root=name+"_target.jpeg")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_batch_size = 8*8

    test_loader = preload.dataloader.DataLoader(
        preload.datasets.MNISTDataset('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=test_batch_size)

    x, y = test_loader[0]
    x = x.to(device).split(8)[0]
    y = y.to(device).split(8)[0]
    x_t, y_t = test_loader[1]
    x_t = x_t.to(device).split(8)[0]
    y_t = y_t.to(device).split(8)[0]

    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    adv_saliency_evaluate(model, device, x, y, x_t, y_t, "Normal")

    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_cnn_adv_guided.pt"))
    adv_saliency_evaluate(model, device, x, y, x_t, y_t, "AGT")

    # intprtr = GRAD(model).to(device)
    # x, y = test_loader[0]
    # x = x.to(device).split(8)[0]
    # y = y.to(device).split(8)[0]
    # x_t, y_t = test_loader[1]
    # x_t = x_t.to(device).split(8)[0]
    # y_t = y_t.to(device).split(8)[0]
    # saliency(intprtr, x, y, root="Normal_normal.jpeg")
    # fgsm = FastGradientSignMethod(lf=F.nll_loss, eps=0.67)
    # x_adv = fgsm.generate(model, x, y_t)
    # saliency(intprtr, x_adv, y_t, root="Normal_adv.jpeg")
    # saliency(intprtr, x_t, y_t, root="Normal_target.jpeg")

    # saliency_map = intprtr(x, y)
    # saliency_map = saliency_map.split(8)[0]
    # for i in range(len(saliency_map)):
    #     max, min = saliency_map[i].max(), saliency_map[i].min()
    #     saliency_map[i] = (saliency_map[i] - min) / (max - min)
    # image = x.split(8)[0]
    # image = torch.cat((image, saliency_map))
    # fp = "./val.jpeg"
    # torchvision.utils.save_image(image, fp)



if __name__ == '__main__':
    main()