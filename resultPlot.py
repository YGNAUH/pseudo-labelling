from UnetFile import DoubleConv, UnetModel
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import random

num_folder_train = 14
num_folder_valid = 3
idx_folder_test = num_folder_train + num_folder_valid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class OCT_BScan_test(Dataset):

    def __init__(self, input_imgs_path, segmented_imgs_path):
        super().__init__()

        self.segDir = segmented_imgs_path
        self.inputDir = input_imgs_path
        self.segList =  os.listdir(self.segDir)
        self.inputList = os.listdir(self.inputDir)

    def __len__(self):

        length = len(self.inputList)

        return length

    def __getitem__(self, idx):

        dirIn = os.path.join(self.inputDir,self.inputList[idx])
        dirSeg = os.path.join(self.segDir,self.segList[idx])

        imgIn = Image.open(dirIn).convert('L')
        imgSeg = Image.open(dirSeg).convert('L')
        imgIn = transforms.Resize((512,1024))(imgIn)
        imgSeg = transforms.Resize((512,1024))(imgSeg)

        ct = transforms.ToTensor()
        input_image = ct(imgIn)
        target_image = ct(imgSeg)


        return (input_image, target_image)


label_path = "./plot_label/"
original_path = "./plot_ori/"



testSet = OCT_BScan_test(original_path,label_path)

testLoader = DataLoader(testSet,batch_size=1)
print(len(testSet), "test images prepared")

def mask2graph(labels):
  graph_output = torch.where(labels > 0.5, torch.tensor(0.0392),
   torch.tensor(0.0))
  return graph_output

def binaryMask(labels):
  graph_output = torch.where((labels > 0.5), torch.tensor(1),
        torch.tensor(0))
  return graph_output


def graph2mask(graph_input):
  label_output = torch.where((graph_input > 0.00) & (graph_input < 0.0392),
        torch.tensor(1), torch.tensor(0))
  return label_output

def lineDrawingPred(predicted_labels):
  # input dimension (h,w)
  h, w = predicted_labels.shape
  #print(h,w)
  idx_h = []
  idx_w = []
  for i in range(1,h):
    for j in range(w):
      if predicted_labels[i-1][j] != predicted_labels[i][j]:
        idx_h.append(i+2)
        idx_h.append(i+1)
        idx_h.append(i)
        idx_h.append(i-1)
        idx_h.append(i-2)

        idx_w.append(j)
        idx_w.append(j)
        idx_w.append(j)
        idx_w.append(j)
        idx_w.append(j)
  result = torch.zeros(3,h,w)
  for i in range(len(idx_h)):
    result[0][idx_h[i]][idx_w[i]] = 1
    result[1][idx_h[i]][idx_w[i]] = 1
    result[2][idx_h[i]][idx_w[i]] = 0



  return result

def lineDrawingLabel(labels):
  # input dimension (n*h,w)
  h, w = labels.shape
  #print(h,w)
  idx_h = []
  idx_w = []
  for i in range(1,h):
    for j in range(w):
      if labels[i-1][j] != labels[i][j]:
        idx_h.append(i+2)
        idx_h.append(i+1)
        idx_h.append(i)
        idx_h.append(i-1)
        idx_h.append(i-2)


        idx_w.append(j)
        idx_w.append(j)
        idx_w.append(j)
        idx_w.append(j)
        idx_w.append(j)
  result = torch.zeros(3,h,w)
  for i in range(len(idx_h)):
    result[0][idx_h[i]][idx_w[i]] = 0
    result[1][idx_h[i]][idx_w[i]] = 1
    result[2][idx_h[i]][idx_w[i]] = 0


  return result


def mask_test(graph_input):
  label_output = torch.where((graph_input > 0.00) & (graph_input < 0.0157),
        torch.tensor(1), torch.tensor(0))
  return label_output


# This function changes bi

def RGBresult(index):
    for i, batch in enumerate(testLoader):
        image = batch[0].to(device)
        target_graph = batch[1]
        label = batch[1].view(512,1024)
        print(torch.unique(label))
        label = mask2graph(mask_test(label))

        pred_label = model(image).view(512,1024).cpu()


        pred = mask2graph(binaryMask(pred_label))
        predLines = lineDrawingPred(pred)
        labelLines = lineDrawingLabel(label)
        background = image.view(512,1024).cpu()

        trans1 = transforms.ToPILImage()

        background = trans1(background).convert("RGBA")
        overlayPred = trans1(predLines).convert("RGBA")
        overlayLabel = trans1(labelLines).convert("RGBA")
        overlay = Image.blend(overlayPred, overlayLabel, 0.5)
        new_img = Image.blend(background, overlay, 0.6)

        new_img.save(index+str(i)+".tif","TIFF")
        break

    #save_name = "./pseudo_exp/5ratio0/"+"result" + str(index) + "tif"
    #save_name = "./result" + str(index) + ".tif"
    #result.save(save_name,"TIFF")



model = UnetModel(1,1).to(device)
#model_file = "./pseudo.pt"
model_file = "./set5.pt"
#model_file = "./Unet_fullES.pt"
model.load_state_dict(torch.load(model_file))
RGBresult("set5")
