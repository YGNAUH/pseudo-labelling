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
import xlsxwriter


# Paramter used here:
num_epoch = 10
num_augmentation = 1
pseudo_ratio = 0

num_labelfolder = 3
num_pseudofolder = 1
num_validfolder = 1
num_testfolder = 1

earlyStoppingFlag = True
patience = 10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def graph2mask(graph_input):
  label_output = torch.where((graph_input > 0.00) & (graph_input < 0.004),
        torch.tensor(1.), graph_input)
  label_output = torch.where((label_output > 0.00) & (label_output != 1.),
        torch.tensor(2.), label_output)
    
  return label_output


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


        imgIn = imgIn.resize((384,896),Image.NEAREST)
        imgSeg = imgSeg.resize((384,896), Image.NEAREST)
        
        ct = transforms.ToTensor()
        input_image = ct(imgIn)
        target_image = ct(imgSeg)

        return input_image, target_image

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

def loss_function(input,target):
    weight_CE = torch.FloatTensor([1,20,1]).to(device)
    lossFcn = nn.CrossEntropyLoss(weight = weight_CE)
    #lossFcn = nn.CrossEntropyLoss()
    loss = lossFcn(input,target)

    return loss

def loss_function_pseudo(input,target):
    weight_CE = torch.FloatTensor([1,5,1]).to(device)
    lossFcn = nn.CrossEntropyLoss(weight = weight_CE)
    #lossFcn = nn.CrossEntropyLoss()
    loss = lossFcn(input,target)

    return loss


def needlemask(label_input):
  label_output = torch.where((label_input != 1),
        torch.tensor(0).int(), label_input)
  return label_output

def tissuemask(label_input):
  label_output = torch.where((label_input < 2),
        torch.tensor(0).int(), label_input)
  label_output = torch.where((label_output > 0),
        torch.tensor(1).int(), label_output)
  return label_output

def bgmask(label_input):
  label_output = torch.where((label_input > 0),
        torch.tensor(2).int(), label_input)
  label_output = torch.where((label_output < 2),
        torch.tensor(1).int(), label_output)
  return label_output

def iou_tissue(outputs, labels):

    outputs = outputs.squeeze(1)
    outputs = tissuemask(outputs)
    labels = tissuemask(labels)

    intersection = (outputs & labels).float().sum((1, 2)) 
    union = (outputs | labels).float().sum((1, 2))        

    iou = (intersection/union )  
    if torch.isnan(iou):
        return 1
    else:
        return iou

def iou_bg(outputs, labels):
    outputs = outputs.squeeze(1) 
    outputs = bgmask(outputs)
    labels = bgmask(labels)

    intersection = (outputs & labels).float().sum((1, 2)) 
    union = (outputs | labels).float().sum((1, 2))        

    iou = (intersection/union )  
    if torch.isnan(iou):
        return 1
    else:
        return iou

def iou_fcn(outputs, labels):
    # BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    #labels = labels.squeeze(1)

    outputs = needlemask(outputs)
    labels = needlemask(labels)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection/union )  # We smooth our devision to avoid 0/0
    if torch.isnan(iou):
        return 1

    else:
        return iou



label_path = "./labels3/"
original_path = "./ori/"
labelled_image_file = sorted(os.listdir(label_path))
original_image_file = sorted(os.listdir(original_path))
#print(labelled_image_file)
end_label = 3
for i in range(end_label):
    print(labelled_image_file[i],original_image_file[i])
    if labelled_image_file[i]!= original_image_file[i]:
        break
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + original_image_file[i])
    len_folder = len(os.listdir(label_folder_path))
    if i == 0:
        scans = OCT_BScan(original_folder_path,label_folder_path)
        for ii in range(num_augmentation):
            scans += OCT_BScan_AU(original_folder_path,label_folder_path)

    else:
        scans += OCT_BScan(original_folder_path,label_folder_path)
        for i in range(num_augmentation):
            scans += OCT_BScan_AU(original_folder_path,label_folder_path)

trainLoader = DataLoader(scans,batch_size=1)
print("total images for training",len(scans))

end_pseudo = 6
for i in range(end_label,end_pseudo):
    print(labelled_image_file[i],original_image_file[i])
    if labelled_image_file[i]!= original_image_file[i]:
        break
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + original_image_file[i])
    if i == end_label:
        pseudoSet = OCT_BScan(original_folder_path,label_folder_path)
    else:
        pseudoSet += OCT_BScan(original_folder_path,label_folder_path)

pseudoLoader = DataLoader(pseudoSet, batch_size=1)
pseudolabels = torch.zeros(len(pseudoSet),1,896,384)
print("total images for pseudolabels:",len(pseudoSet))

end_vali = 7
for i in range(end_pseudo,end_vali):
    print(labelled_image_file[i],original_image_file[i])
    if labelled_image_file[i]!= original_image_file[i]:
        break
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + original_image_file[i])
    validSet = OCT_BScan(original_folder_path,label_folder_path)
validLoader = DataLoader(validSet,batch_size=1)
print("total images for validation:",len(validSet))


model_label = UnetModel(1,3).to(device)
optimizer_label = optim.Adam(model_label.parameters(), lr=1e-3)
scheduler_label = optim.lr_scheduler.StepLR(optimizer_label, step_size=20, gamma=0.1)
#scheduler_label = optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, "max", patience = 5)

def reward_process(model):
    count = 0
    iou_needle = 0
    iou_ti= 0
    iou_back=0
    reward = 0
    for i, batch in enumerate(trainLoader):
        input_graph = batch[0].to(device)

        target_graph = batch[1][0]
        target_label = graph2mask(target_graph).int()

        pred_label = model(input_graph).cpu()
        predictions = torch.nn.functional.softmax(pred_label, dim=1)
        pred = torch.argmax(predictions, dim=1) 
        pred = pred.int()
        uni, count_table = torch.unique(pred, return_counts=True,sorted=True)

        iou = iou_fcn(pred, target_label)
        #print(len(torch.unique(target_label)))
        if len(torch.unique(target_label)) == 3:
            iou_tmp = iou_fcn(pred, target_label)
            iou_needle += iou_tmp
            
            if iou_tmp > 0.5:
                reward += 1
            count += 1
            print(iou_tmp, count_table,reward)

        iou_ti += iou_tissue(pred, target_label)
        iou_back += iou_bg(pred, target_label)

        
    iou = [iou_needle/count, iou_ti/len(scans), iou_back/len(scans)]

    flag = False
    if reward/len(scans) > 0.5:
        flag = True
    
        

    return iou, flag



def valid_process(model):
    count = 0
    iou_needle = 0
    iou_back = 0
    iou_ti = 0
    for i, batch in enumerate(validLoader):
        input_graph = batch[0].to(device)

        target_graph = batch[1][0]
        target_label = graph2mask(target_graph).int()

        pred_label = model(input_graph).cpu()
        predictions = torch.nn.functional.softmax(pred_label, dim=1)
        pred = torch.argmax(predictions, dim=1) 
        pred = pred.int()
        uni, count_table = torch.unique(pred, return_counts=True,sorted=True)

        iou = iou_fcn(pred, target_label)
        #print(len(torch.unique(target_label)))
        if len(torch.unique(target_label)) == 3:
            iou_tmp = iou_fcn(pred, target_label)
            iou_needle += iou_tmp
            print(iou_tmp, count_table)
            count += 1

        iou_ti += iou_tissue(pred, target_label)
        iou_back += iou_bg(pred, target_label)

        
    iou = [iou_needle/count, iou_ti/len(validSet), iou_back/len(validSet)]
        

    return iou


#################################3  Training Process  ############################################3:
valid_record = []
print("start training")
for epoch in range(num_epoch):
  running_loss = 0
  
  for i, batch in enumerate(trainLoader):
      input_graph = batch[0].to(device)
      target_graph = batch[1][0]
      target_label = graph2mask(target_graph).to(device)
      #print(torch.unique(target_label))

      optimizer_label.zero_grad()
      output_label = model_label(input_graph).to(device)

      loss = loss_function(output_label.to(torch.float32),target_label.long())
      loss.backward()
      optimizer_label.step()
      running_loss += loss.item()

  scheduler_label.step()
 
  model1_loss = running_loss/len(scans)
  print("epoch", epoch, "loss: ",  model1_loss)
  iou, flag = reward_process(model_label)

  print("epoch", epoch, "iou: ", iou)
  valid_record.append(iou[0])
  if iou[0] > 0.6 or flag:
    print("model firstly trained, 60% IOU achieved")
    break
'''
###### predictions #######:
for i, batch in enumerate(pseudoLoader):
      input_graph = batch[0].to(device)
      pred_label = model_label(input_graph).cpu()
      predictions = torch.nn.functional.softmax(pred_label, dim=1)
      pred = torch.argmax(predictions, dim=1) 
      pred = pred.int()
      pseudolabels[i] = pred 
print("prediction finished and start to train again")


###### training with pseudo-labels ###################:
best_valid_iou = 0
triggered = 0
for epoch in range(num_epoch):
  running_loss = 0
  for i, batch in enumerate(trainLoader):
      input_graph = batch[0].to(device)
      target_graph = batch[1][0]
      target_label = graph2mask(target_graph).to(device)

      optimizer_label.zero_grad()
      output_label = model_label(input_graph).to(device)

      loss = loss_function(output_label.to(torch.float32),target_label.long())
      loss.backward()
      optimizer_label.step()
      running_loss += loss.item()
  model1_loss = running_loss/len(scans)
  print("epoch", epoch, "label loss: ",  model1_loss)

  running_loss = 0
  for i, batch in enumerate(pseudoLoader):
      input_graph = batch[0].to(device)
      target_label = pseudolabels[i].to(device)

      optimizer_label.zero_grad()
      output_label = model_label(input_graph).to(device)

      loss = 0.5*loss_function_pseudo(output_label.to(torch.float32),target_label.long())
      loss.backward()
      optimizer_label.step()
      running_loss += loss.item()
  model1_loss = running_loss/len(scans)
  print("epoch", epoch, "pseudo loss: ",  model1_loss)

  #update:
  for i, batch in enumerate(pseudoLoader):
      input_graph = batch[0].to(device)
      pred_label = model_label(input_graph).cpu()
      predictions = torch.nn.functional.softmax(pred_label, dim=1)
      pred = torch.argmax(predictions, dim=1) 
      pred = pred.int()
      pseudolabels[i] = pred 

  iou = valid_process(model_label)
  valid_record.append(iou[0])
  print(epoch, iou)

  
  if (earlyStoppingFlag):
    if iou[0] > best_valid_iou:
        best_valid_iou = iou[0]
        triggered = 0
    else:
        triggered += 1

        print("triggered", triggered)

        if (triggered > patience):
                print("early stopping")
                break

'''
save_path = "./"
model_name = "pse3full.pt"
torch.save(model_label.state_dict(), save_path + model_name)
print("model saved")
print(valid_record)
