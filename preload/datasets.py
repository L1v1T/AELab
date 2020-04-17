import torch
from torchvision import datasets, transforms

from PIL import Image


class MNISTDataset(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNISTDataset, self).__init__(root, train=train, transform=transform,
                                    target_transform=target_transform, download=download)
        self.examples = []
        for index in range(len(self.data)):
            img, target = self.data[index], int(self.targets[index])
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            self.examples.append((img, target))
        

    def __getitem__(self, index):
        return self.examples[index]