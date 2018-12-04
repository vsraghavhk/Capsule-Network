from model_base import ModelBase
from model import ModelIdol
import numpy as np 
import cv2
import os

def obtain():
    ckpt = "outputs/checkpoints/Idols--1523697220.148633" #c1s_9_c1n_256_c2s_6_c2n_64_c2d_0.7_c1vl_16_c1s_5_c1nf_16_c2vl_32_lr_0.0001_rs_1--
    #model = ModelIdol("Idols", output_folder=None)
    # Load the model
    #model.load(ckpt)
    
    result = test(ckpt)
    return result[1]

def test(ckpt):
    model = ModelIdol("Idols", output_folder=None)
    # Load the model
    model.load(ckpt)
    folder = "test"
    images = []
    labels = []
    for filename in os.listdir (folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append('0')
    # Indian Jugaad
    images = np.array(images) 
    labels = np.array(labels)

    prediction = model.predict_image(images, labels)
    #'''
    with open("signnames.csv", "r") as f:
        signnames = f.read()
    id_to_name = { int(line.split(",")[0]):line.split(",")[1] for line in signnames.split("\n")[1:] if len(line) > 0}
    if prediction[0] in id_to_name:
        string = id_to_name[prediction[0]]
    print("The image is : ", string)
    #'''
    return prediction

if __name__ == '__main__':
    obtain()