import os
import cv2


def read_image(filename):
    image = cv2.imread(filename)
    return cv2.resize(image, (64,64))

def write_images(images, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(images.shape[0]):
        cv2.imwrite("%s/sample_%.6d.png" %(folder, i), images[i])
