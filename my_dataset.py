from PIL import Image
import torch
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform = None, target_transform = None):
        super(MyDataset, self).__init__()
        labels_fo = open(root + "/labels.txt", mode = 'r')
        items = []
        for line in labels_fo:
            line = line.rstrip('\n')
            words = line.split('\t')
            items.append((words[0], int(words[1])))
        self.items = items
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fid, label = self.items[index]
        img = Image.open(self.root + "/" + fid + ".png").convert('L')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.items)