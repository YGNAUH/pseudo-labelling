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
num_epoch = 20
num_folder_train = 14
num_folder_valid = 3


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


label_path = "./labelled_OCT/"
original_path = "./original_OCT/"
labelled_image_file = os.listdir(label_path)
original_image_file = os.listdir(original_path)

for i in range(num_folder_train):
  label_folder_path = os.path.join(label_path + labelled_image_file[i])
  original_folder_path = os.path.join(original_path + labelled_image_file[i])
  if i == 0:
    trainSet = OCT_BScan(original_folder_path,label_folder_path)
  else:
    trainSet += OCT_BScan(original_folder_path,label_folder_path)
print("total images for trainning:",len(trainSet))
trainLoader = DataLoader(trainSet,batch_size=1)


for i in range(num_folder_train, num_folder_train+num_folder_valid):
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + labelled_image_file[i])
    if i == num_folder_train:
      validSet = OCT_BScan(original_folder_path,label_folder_path)
    else:
      validSet += OCT_BScan(original_folder_path,label_folder_path)
validLoader = DataLoader(validSet,batch_size=1)



def test4DataLoader():
  for i, batch in enumerate(trainLoader):
    if i == 0:
      input_graph = batch[0]
      target_graph = batch[1]
      print(input_graph.shape)
      print(target_graph.shape)
      target_stack = target_graph
      break
#test4DataLoader()

model = UnetModel(1,1).to(device)
# Training for full resolution
optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter("./log")

#Training Process:
epoch_loss_list = []
epoch_valid_acc_list = []
epoch_valid_iou_list = []
epoch_valid_loss_list = []
SMOOTH = 1e-6

def iou_fcn(outputs, labels):
    # BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded

def acc_fcn(outputs, labels):
    correct = (outputs == labels).float()
    acc = correct.sum()/correct.numel()
    return acc

print("start trainning")
start_time = time()
triggered = 0
patience = 2
best_valid_acc = 0
best_valid_iou = 0
# Trainning Loop
for epoch in range(num_epoch):

    running_loss = 0
    for i, batch in enumerate(trainLoader):
        input_graph = batch[0].to(device)
        target_graph = batch[1]
        target_label = graph2mask(target_graph).to(device)

        optimizer.zero_grad()
        output_label = model(input_graph).to(device)

        loss = loss_function(output_label.to(torch.float32),target_label.to(torch.float32))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss/len(trainSet)
    print(epoch,"train loss", epoch_loss)
    # validation:
    running_acc = 0
    running_iou = 0
    valid_loss = 0
    for i, batch in enumerate(validLoader):
        input_graph = batch[0].to(device)
        label = graph2mask(batch[1])
        pred_label = model(input_graph).cpu()
        loss = loss_function(pred_label.to(torch.float32),
                                label.to(torch.float32))
        valid_loss += loss.item()
        pred = binaryMask(pred_label)
        running_iou += float(iou_fcn(pred,label)[0])
        running_acc += float(acc_fcn(pred,label))
    epoch_valid_iou = running_iou/len(validSet)
    epoch_valid_acc = running_acc/len(validSet)
    epoch_valid_loss = valid_loss/len(validSet)

    epoch_loss_list.append(epoch_loss)
    epoch_valid_loss_list.append(epoch_valid_loss)
    epoch_valid_acc_list.append(epoch_valid_acc)
    epoch_valid_iou_list.append(epoch_valid_iou)


    writer.add_scalar('train loss/epoch', epoch_loss, epoch)
    writer.add_scalar('valid loss/epoch', epoch_valid_loss, epoch)
    writer.add_scalar('acc/epoch', epoch_valid_acc, epoch)
    writer.add_scalar('iou/epoch', epoch_valid_iou, epoch)
    print(epoch, "loss, acc, iou",
            epoch_valid_loss, epoch_valid_acc, epoch_valid_iou)

    if epoch_valid_acc > best_valid_acc:
        best_valid_acc = epoch_valid_acc
        if epoch_valid_iou > best_valid_iou:
            best_valid_iou = epoch_valid_iou
            triggered = 0
        else:
            triggered += 1
            print("triggered")
    else:
            triggered += 1
            print("triggered")

    if (triggered > patience):
            print("early stopping")
            break
end_time = time()
seconds_elapsed = end_time - start_time
mins, rest = divmod(seconds_elapsed, 60)
mins = str(mins).split(".")
rest = str(rest).split(".")

time_used =  mins[0] + "." + rest[0]
#Writing Result to Excel
print('Training done and writing result')

def write_save():
  workbook = xlsxwriter.Workbook("fullTrainExcel.xlsx")
  worksheet = workbook.add_worksheet()

  row_numEpoc = 0
  row_time = 1
  row_epochLoss = 2
  row_validLoss = 3
  row_iou = 4
  row_acc = 5

  worksheet.write(0, row_numEpoc, "epoch")
  worksheet.write(1, row_numEpoc, num_epoch)
  worksheet.write(0, row_time, "time")
  worksheet.write(1, row_time, time_used)
  worksheet.write(0, row_epochLoss , "train loss")
  worksheet.write(0, row_validLoss , "valid loss")
  worksheet.write(0, row_iou , "IOU")
  worksheet.write(0, row_acc , "accuracy")


  for i in range(len(epoch_loss_list)):
      tl = str(epoch_loss_list[i]).split(".")
      vl = str(epoch_valid_loss_list[i]).split(".")
      iou = str(epoch_valid_iou_list[i]).split(".")
      acc = str(epoch_valid_acc_list[i]).split(".")

      if len(tl[1])>5:
          tl[1] = tl[1][0:5]
      epoch_loss_list[i] = float((".").join(tl))
      if len(vl[1])>5:
          vl[1] = vl[1][0:5]
      epoch_valid_loss_list[i] = float((".").join(vl))

      if len(iou[1])>5:
          iou[1] = iou[1][0:5]
      epoch_valid_iou_list[i] = float((".").join(iou))
      if len(acc[1])>5:
          acc[1] = acc[1][0:5]
      epoch_valid_acc_list[i] = float((".").join(acc))

  print(epoch_loss_list,epoch_valid_acc_list,epoch_valid_loss_list)
  worksheet.write_column(1,row_epochLoss,epoch_loss_list)
  worksheet.write_column(1,row_validLoss,epoch_valid_loss_list)
  worksheet.write_column(1,row_iou,epoch_valid_iou_list)
  worksheet.write_column(1,row_acc,epoch_valid_acc_list)
  workbook.close()
  print('Finished writing result and saving model')

  # save trained model
  torch.save(model.state_dict(), 'Unet_fullES.pt')
  print('Model saved.')

write_save()
