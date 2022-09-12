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
num_epoch = 40
num_folder_train = 1
num_folder_valid = 0


num_folder_label = 1

num_augmentation = 0
pseudo_ratio = 0


earlyStoppingFlag = True
patience = 20
epoch_label = 0
SMOOTH = 1e-6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# This function gives binary mask to input graph
#background with label 0 and segmented-layers with label 1
def graph2mask(graph_input):
  label_output = torch.where((graph_input > 0.00) & (graph_input < 0.0392),
        torch.tensor(1), torch.tensor(0))
  return label_output

# This function changes binary mask to color (0 for 0, 0.0392 for 1)
def mask2graph(labels):
  graph_output = torch.where(labels > 0.5, torch.tensor(0.0392), torch.tensor(0.))
  return graph_output

def binaryMask(labels):
  graph_output = torch.where((labels > 0.5), torch.tensor(1),
        torch.tensor(0))
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


        imgIn = transforms.Resize((896,384))(imgIn)
        imgSeg = transforms.Resize((896,384))(imgSeg)
        ct = transforms.ToTensor()
        input_image = ct(imgIn)
        target_image = ct(imgSeg)

        return input_image, target_image


label_path = "./labels2/"
original_path = "./ori/"
#labelled_image_file = os.listdir(label_path)
original_image_file = os.listdir(original_path)


label_folder_path = os.path.join(label_path + "0_03_28-RIAZ-3-17")
original_folder_path = os.path.join(original_path + original_image_file[0])
len_folder = len(os.listdir(label_folder_path))
scans = OCT_BScan(original_folder_path,label_folder_path)
trainLoader = DataLoader(scans,batch_size=1)


SMOOTH = 1e-6

def iou_fcn(outputs: np.array, labels: np.array):
    #outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = intersection  / union 
  

    
    return iou # Or thresholded.mean()

def demo_iou():
  iou_sum = 0
  num_needle = 0
  for i, batch in enumerate(trainLoader):
    if (i+1)%8 == 0:
      input_graph = batch[0].to(device)

      target_graph = batch[1][0]
      #print(target_graph.shape)
      npa = np.array(graph2mask(target_graph))
      
      pred_label = model(input_graph).cpu()

      
      pred = binaryMask(pred_label)
      pred = np.array(pred)[0]

      iou = iou_fcn(pred, npa)

      if len(np.unique(target_graph)) == 1:
        pass
      else:
        iou_sum += iou
        num_needle += 1
      #print(pred.shape)
      #print(np.unique(pred), np.unique(npa))
      print(i, " iou ",iou)
  print(iou_sum, num_needle)
  iou = iou_sum/num_needle
  print("average iou: ", iou)
    

def demo():
  final_image = Image.new("RGBA",(384*6,896))
  idx = 0

  for i_batch, batch in enumerate(trainLoader):
    if (i_batch+1 )%8 == 0:
      input_graph = batch[0].to(device)
      target_graph = batch[1][0]
      #print(target_graph.shape)
      npa = np.array(target_graph)
      print(npa.shape)

      pred_label = model(input_graph).cpu()
      pred = mask2graph(binaryMask(pred_label))
      pred = np.array(pred)
      #print(pred.shape)

      input_graph = input_graph[0].cpu()
      pred = pred[0]
      #print(input_graph.shape)

      # save original image
      trans1 = transforms.ToPILImage()
      background = trans1(input_graph).convert("L")
      #background.save("./bg.tif")

      img_np = npa[0]

      h = img_np.shape[0]
      w = img_np.shape[1]
      print(i_batch,"h, w",h,w)

      #save segmentation
      result = torch.zeros(3,h,w)
      for i in range(h):
          for j in range(w):
              if img_np[i][j] == 0:
                  result[0][i][j] = 1
                  result[1][i][j] = 1
                  result[2][i][j] = 1
              else:
                  result[0][i][j] = 1
                  result[1][i][j] = 0
                  result[2][i][j] = 1

      trans1 = transforms.ToPILImage()

      result = trans1(result).convert("RGBA")
      #result.save("./segmentation.tif","TIFF")
 
      pred_result= torch.zeros(3,h,w)
      pred = pred[0]
      for i in range(h):
          for j in range(w):
              if pred[i][j] == 0:
                  pred_result[0][i][j] = 1
                  pred_result[1][i][j] = 1
                  pred_result[2][i][j] = 1
              else:
                  pred_result[0][i][j] = 0
                  pred_result[1][i][j] = 1
                  pred_result[2][i][j] = 0

      trans1 = transforms.ToPILImage()
      pred_result = pred_result
      result = result

      pred_result = trans1(pred_result).convert("RGBA")
      #pred_result.save("./pred.tif","TIFF")  

      overlay = Image.blend(result, pred_result, 0.5)
      final_image.paste(overlay, box = (384*idx, 896))
      overlay_name = "./over" + str(i_batch) +".tif"
      #overlay = overlay.rotate(-90, expand=True)
      overlay.save(overlay_name,"TIFF")
      idx += 1
  #final_image.save("./final.tif","TIFF")
      #overlay.save("./mixed.tif","TIFF")





model = UnetModel(1,1).to(device)
#model_file = "./pseudo_exp/pse_AU/set" + str(i) + ".pt"
#model_file = "./model2class/bi0328.pt"
model_file = "./testModel.pt"
model.load_state_dict(torch.load(model_file))
demo_iou()
demo()