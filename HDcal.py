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
import scipy.spatial

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


label_path = "./labelled_OCT/"
original_path = "./original_OCT/"
labelled_image_file = os.listdir(label_path)
original_image_file = os.listdir(original_path)


for i in range(idx_folder_test, len(labelled_image_file )):
    print(labelled_image_file[i])
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + labelled_image_file[i])
    if i == idx_folder_test:
      testSet = OCT_BScan_test(original_folder_path,label_folder_path)
    else:
      testSet = OCT_BScan_test(original_folder_path,label_folder_path)


testLoader = DataLoader(testSet ,batch_size=1)
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





def HD_measure(model):
  hd_pse = 0
  hd_f = 0
  for i, batch in enumerate(testLoader):
  
      #model_file = "./pseudo.pt"
      #model.load_state_dict(torch.load(model_file))
      image = batch[0].to(device)
      target_graph = batch[1]
      label = batch[1].view(512,1024)
      #pred_label = model(image).view(512,1024).cpu()
      #pred = mask2graph(binaryMask(pred_label))
      #hd = scipy.spatial.distance.directed_hausdorff(pred, label)
      #print("Pseudo-labelling HD: ")
      #print(hd[0])
      #hd_pse += hd[0]
      
      #model_file = "./fsl.pt"
      #model_file = "./Unet_fullES.pt"
      #model.load_state_dict(torch.load(model_file))
      pred_label = model(image).view(512,1024).cpu()
      pred = mask2graph(binaryMask(pred_label))
      hd = scipy.spatial.distance.directed_hausdorff(pred, label)
      hd_f += hd[0]


      #print("Fully Supervised HD: ")
      #print(hd[0])
  return hd_f/len(testSet)
#print("Pseudo-labelling HD: ", hd_pse/len(testSet))
#print("Fully Supervised HD: ",hd_f/len(testSet))

hd_list = []
for i in range(5,10):
  model = UnetModel(1,1).to(device)
  model_file = "./pseudo_exp/5ratio0/set" + str(i) + ".pt"
  model.load_state_dict(torch.load(model_file))
  hd = HD_measure(model)
  hd_list.append(hd)
  print(i, "HD: ",hd)

print(hd_list.sum()/len(hd_list))