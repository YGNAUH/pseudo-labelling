import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import random
from UnetFile import DoubleConv, UnetModel
from torch.utils.tensorboard import SummaryWriter
from time import time



# Paramter used here:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Using device:', device)

def graph2mask(graph_input):
  label_output = torch.where((graph_input > 0.00) & (graph_input < 0.004),
        torch.tensor(1.), graph_input)
  label_output = torch.where((label_output > 0.00) & (label_output != 1.),
        torch.tensor(2.), label_output)
    
  return label_output

# This function changes binary mask to color (0 for 0, 0.0392 for 1)
def mask2graph(labels):
  graph_output = torch.where(labels < 0.5, torch.tensor(0.0), labels)
  graph_output = torch.where(labels > 1.5, torch.tensor(0.00784), graph_output)
  graph_output = torch.where((graph_output > 0.5), torch.tensor(0.00392),graph_output)


  return graph_output

def class3Mask(labels):
  graph_output = torch.where((labels > 0.75) & (labels < 1.25), torch.tensor(1.0),labels)
  graph_output = torch.where((labels < 0.75), torch.tensor(0.),graph_output)
  graph_output = torch.where((labels > 1.25), torch.tensor(2.),graph_output)


  return graph_output

def augFcn(imgIn, tarIn):
  angle = random.randint(5, 10)
  brightness_factor = random.uniform(1,2)
  #print(angle, brightness_factor)

  input_image = transforms.functional.rotate(imgIn, angle)
  input_image = transforms.functional.adjust_brightness(input_image, brightness_factor)
  target_image = transforms.functional.rotate(tarIn, angle)
  target_image = transforms.functional.adjust_brightness(target_image, brightness_factor)
  return input_image,target_image


class OCT_BScan_AU(Dataset):

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

        imgIn = imgIn.resize((384,896),Image.NEAREST)
        imgSeg = imgSeg.resize((384,896), Image.NEAREST)
        
        ct = transforms.ToTensor()
        imgIn, imgSeg = augFcn(imgIn, imgSeg)
        input_image = ct(imgIn)
        target_image = ct(imgSeg)

        return input_image, target_image


class OCT_BScan(Dataset):

    def __init__(self, input_imgs_path, segmented_imgs_path):
        super().__init__()

        self.segDir = segmented_imgs_path
        self.inputDir = input_imgs_path
        self.segList =  sorted(os.listdir(self.segDir))
        self.inputList = sorted(os.listdir(self.inputDir))

    def __len__(self):

        length = len(self.inputList)

        return length

    def __getitem__(self, idx):

        dirIn = os.path.join(self.inputDir,self.inputList[idx])
        dirSeg = os.path.join(self.segDir,self.segList[idx])

        imgIn = Image.open(dirIn).convert('L')
        imgSeg = Image.open(dirSeg).convert('L')


        imgIn = imgIn.resize((384,896),Image.NEAREST)
        imgSeg = imgSeg.resize((384,896), Image.NEAREST)
        
        ct = transforms.ToTensor()
        input_image = ct(imgIn)
        target_image = ct(imgSeg)

        return input_image, target_image



label_path = "./labels3/"
original_path = "./ori/"
labelled_image_file = sorted(os.listdir(label_path))
original_image_file = sorted(os.listdir(original_path))
print(labelled_image_file, original_image_file)

num_augmentation = 0

for i in range(6,7):
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + original_image_file[i])
    len_folder = len(os.listdir(label_folder_path))
    if i == 0:
        scans = OCT_BScan(original_folder_path,label_folder_path)
        for ii in range(num_augmentation):
            if ii == 0:
                aug_scans = OCT_BScan_AU(original_folder_path,label_folder_path)
            else:
                aug_scans += OCT_BScan_AU(original_folder_path,label_folder_path)

    else:
        scans += OCT_BScan(original_folder_path,label_folder_path)
        for i in range(num_augmentation):
            aug_scans += OCT_BScan_AU(original_folder_path,label_folder_path)

trainLoader = DataLoader(scans,batch_size=1)
print("total images for training",len(scans))



def demo():
  for idx, batch in enumerate(trainLoader):
      input_graph = batch[0].to(device)

      target_graph = batch[1]
      target_label = graph2mask(target_graph)
      print(torch.unique(target_label))

      pred_label = (model(input_graph)).cpu()
      print(pred_label.shape)
      #print(pred_label)
      predictions = torch.nn.functional.softmax(pred_label, dim=1)
      pred = torch.argmax(predictions, dim=1) 
      pred = pred.float()

      
      print(pred.shape)
      print("unique label", torch.unique(target_label))
      
      print("unique pred",torch.unique(pred))
     
      #print("pred shape",pred.shape)

      trans1 = transforms.ToPILImage()

      target_label = target_label[0][0]
      pred = pred[0]
      


      h = pred.shape[0]
      w = pred.shape[1]
      print("h, w",h,w)
      
      # label RGB plot
      result = torch.zeros(3,h,w)
      for i in range(h):
          for j in range(w):
              if target_label[i][j] == 0:
                  result[0][i][j] = 1
                  result[1][i][j] = 1
                  result[2][i][j] = 1
              elif target_label[i][j] == 1:
                  result[0][i][j] = 1
                  result[1][i][j] = 1
                  result[2][i][j] = 0

              else:
                  result[0][i][j] = 1
                  result[1][i][j] = 0
                  result[2][i][j] = 1


      #prediction RGB plot
 
      pred_result= torch.zeros(3,h,w)

      for i in range(h):
          for j in range(w):
              if pred[i][j] == 0:
                  pred_result[0][i][j] = 1
                  pred_result[1][i][j] = 1
                  pred_result[2][i][j] = 1
              elif pred[i][j] == 1:
                  pred_result[0][i][j] = 0
                  pred_result[1][i][j] = 1
                  pred_result[2][i][j] = 0

              else:
                  pred_result[0][i][j] = 1
                  pred_result[1][i][j] = 0
                  pred_result[2][i][j] = 0

      trans1 = transforms.ToPILImage()

      input_image = trans1(input_graph[0].cpu()).convert("RGBA")
      print("INPUT SAVED")
      input_image.save("./res0909/input" + str(idx) + ".tif","TIFF")
      result= trans1(result).convert("RGBA")
      
      pred_result = trans1(pred_result).convert("RGBA")
      #pred_result.save("./pred.tif","TIFF")  

      overlay = Image.blend(result, pred_result, 0.5)
      
      overlay_name = "./res0909/overlay" + str(idx) +".tif"
      result_name = "./res0909/label" + str(idx) +".tif"
      pred_name = "./res0909/pred" + str(idx) +".tif"

      result.save(result_name,"TIFF")
      pred_result.save(pred_name,"TIFF")
      overlay.save(overlay_name,"TIFF")
      print("IMAGES SAVE")
      
  #final_image.save("./final.tif","TIFF")
      #overlay.save("./mixed.tif","TIFF")
   


model = UnetModel(1,3).to(device)
#model_file = "./pseudo_exp/pse_AU/set" + str(i) + ".pt"
#model_file = "./model2class/bi0328.pt"
#model_file = "./testModel.pt"
model_file = "./pse3Model1.pt"
model.load_state_dict(torch.load(model_file))
#demo_iou()
demo()