from attack import attack
from model import Net
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import shutil

ad_train_folder = "ad_train"
ad_test_folder = "ad_test"

def load_model(model_file, device):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

def read_image(file_name):
    image = Image.open(file_name).convert("L")
    trans = transforms.Compose([transforms.ToTensor()])

    image = trans(image)
    image = image.unsqueeze(0)
    return image

def save_image(file_name, image):
    image = image.squeeze(0)
    b=np.array(image)
    maxi=b.max()
    b=b*255./maxi
    b=b.transpose(1,2,0).astype(np.uint8)
    b=np.squeeze(b,axis=2)
    xx=Image.fromarray(b)
    xx.save(file_name)

def trans_dataset(out_folder, in_folder, model):
    for _, _, files in os.walk(in_folder):
        count = 0
        file_list = []
        for fname in files:
            if os.path.splitext(fname)[1] == '.png':
                file_list.append(fname)
            else:
                shutil.copyfile(in_folder + "/" + fname, out_folder + "/" + fname)
        for fimage in file_list:
            image = read_image(in_folder + "/" + fimage)
            new_image = attack(model, image, epsilon = 16/255)
            save_image(out_folder + "/" + fimage, new_image)
            count += 1
            print(count)

def main():
    if not os.path.exists(ad_train_folder):
        os.mkdir(ad_train_folder)
    if not os.path.exists(ad_test_folder):
        os.mkdir(ad_test_folder)

    model = load_model("mnist_cnn.pt", "cpu")
    trans_dataset(ad_train_folder, "train", model)
    trans_dataset(ad_test_folder, "test", model)


if __name__ == "__main__":
    main()