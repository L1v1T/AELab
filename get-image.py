import numpy as np
import os
from PIL import Image

train_images_path = "train-images.idx3-ubyte"
train_labels_path = "train-labels.idx1-ubyte"
test_images_path = "t10k-images.idx3-ubyte"
test_labels_path = "t10k-labels.idx1-ubyte"

train_folder = "train"
test_folder = "test"

def create_folder():
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

def get_images(images_path, labels_path, save_folder):
    image_fo = open(images_path, mode = 'rb')
    if image_fo.closed:
        print("open image ubyte file error.")
        os.abort()
    label_fo = open(labels_path, mode = 'rb')
    if label_fo.closed:
        print("open label ubyte file error.")
        os.abort()

    # read header
    image_fo.read(4)
    img_num = int.from_bytes(image_fo.read(4), byteorder = 'big', signed = False)
    img_row = int.from_bytes(image_fo.read(4), byteorder = 'big', signed = False)
    img_col = int.from_bytes(image_fo.read(4), byteorder = 'big', signed = False)
    label_fo.read(2*4)

    for id in range(img_num):
        # read image
        image_data = np.zeros((img_row, img_col))
        for i in range(img_row):
            for j in range(img_col):
                image_data[i][j] = int.from_bytes(image_fo.read(1), byteorder = 'big', signed = False)

        # read label
        label = int.from_bytes(label_fo.read(1), byteorder = 'big', signed = False)

        # save image and label
        img_path = save_folder + "/" + str(id) + ".png"
        img = Image.fromarray(image_data)
        img = img.convert('RGB')
        img.save(img_path)

        fo = open(save_folder + "/" + "labels.txt", mode = 'a')
        if fo.closed:
            print("open " + save_folder + "/" + "labels.txt error.")
            os.abort()
        fo.write(str(id) + "\t" + str(label) + "\n")
        fo.close()

def main():
    create_folder()

    # create training set
    get_images(train_images_path, train_labels_path, train_folder)

    # create testing set
    get_images(test_images_path, test_labels_path, test_folder)

if __name__ == "__main__":
    main()