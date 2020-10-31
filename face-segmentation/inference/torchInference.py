import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time
import argparse

print(torch.__version__)


parser = argparse.ArgumentParser()
parser.add_argument(
    "model_path", help='Specify the dataset directory path')
parser.add_argument(
    "image_path", help='Specify the experiment directory where metrics and model weights shall be stored.')

args = parser.parse_args()

model_path = args.model_path
image_path = args.image_path

# Load the trained model
if torch.cuda.is_available():
    model = torch.load(model_path)
else:
    model = torch.load(model_path, map_location=torch.device('cpu'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Set the model to evaluate mode
model.eval()

# Read  a sample image and mask from the data-set
originalImage = cv2.imread(image_path)

# Resize image
img = cv2.resize(originalImage, (256, 256), cv2.INTER_AREA).transpose(2,0,1)

# Uncomment above line and use the below one for inference with original image size
# img = originalImage.transpose(2,0,1)

img = img.reshape(1, 3, img.shape[1],img.shape[2])

start_time = time.time()
with torch.no_grad():
    if torch.cuda.is_available():
    	a = model(torch.from_numpy(img).to(device).type(torch.cuda.FloatTensor)/255)
    else:
        a = model(torch.from_numpy(img).to(device).type(torch.FloatTensor)/255)
print("--- %s seconds ---" % (time.time() - start_time))
