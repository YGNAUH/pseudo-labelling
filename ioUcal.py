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

num_folder_train = 14
num_folder_valid = 3
idx_folder_test = num_folder_train + num_folder_valid

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


class OCT_BScan(Dataset):

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

        return input_image, target_image

def loss_function(input,target):

    lossFcn = nn.BCELoss()
    loss = lossFcn(input,target)

    return loss

def iou_fcn(outputs, labels):
    # BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection) / (union)  # We smooth our devision to avoid 0/0

    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou

def acc_fcn(outputs, labels):
    correct = (outputs == labels).float()
    acc = correct.sum()/correct.numel()
    return acc

def mask_test(graph_input):
  label_output = torch.where((graph_input > 0.00) & (graph_input < 0.0157),
        torch.tensor(1), torch.tensor(0))
  return label_output




'''
label_path = "./labelled_OCT/"
original_path = "./original_OCT/"
labelled_image_file = os.listdir(label_path)
original_image_file = os.listdir(original_path)

for i in range( num_folder_train+num_folder_valid, len(labelled_image_file )):
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + labelled_image_file[i])

    if i ==  num_folder_train+num_folder_valid:
        testSet = OCT_BScan(original_folder_path,label_folder_path)
    else:
        testSet += OCT_BScan(original_folder_path,label_folder_path)
testLoader = DataLoader(testSet,batch_size=1)
print("total images for test :",len(testSet))
'''

label_path = "./atest/label/"
original_path = "./atest/ori/"


testSet = OCT_BScan(original_path,label_path)

testLoader = DataLoader(testSet,batch_size=1)
print("total images for test :",len(testSet))

iou_vector = []
acc_vector = []

def test(model_name):
    model = UnetModel(1,1).to(device)
    #model_file = "./Unet_fullES.pt"
    model_file = model_name
    model.load_state_dict(torch.load(model_file))

    test_acc =0
    test_iou =0
    test_loss=0
    for i, batch in enumerate(testLoader):
        input_graph = batch[0].to(device)
        label = mask_test(batch[1])
        pred_label = model(input_graph).cpu()
        loss = loss_function(pred_label.to(torch.float32),
                                label.to(torch.float32))
        test_loss += loss.item()
        pred = binaryMask(pred_label)
        test_iou += float(iou_fcn(pred,label)[0])
        test_acc += float(acc_fcn(pred,label))
    test_acc /= len(testSet)
    test_iou /= len(testSet)
    test_loss /= len(testSet)
    iou_vector.append(test_iou)
    acc_vector.append(test_acc)

for i in range(5,10):
    path = "./pseudo_exp/5ratio0_pre/"
    model_name = "set" + str(i) + ".pt"
    name = path + model_name
    test(name)

    
print(iou_vector,acc_vector)