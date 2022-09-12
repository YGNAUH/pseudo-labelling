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
num_epoch = 40
num_folder_train = 14
num_folder_valid = 3


num_folder_label = 5
image1Folder = 5
num_augmentation = 0
pseudo_ratio = 0

folder_label_range = 10
earlyStoppingFlag = True
patience = 5
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


        imgIn = transforms.Resize((512,1024))(imgIn)
        imgSeg = transforms.Resize((512,1024))(imgSeg)
        ct = transforms.ToTensor()
        imgIn, imgSeg = augFcn(imgIn, imgSeg)
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

def write_result(time_used,epoch_loss_list,model1_iou,
                 epoch_valid_acc_list,epoch_valid_iou_list,epoch_valid_loss_list,
                 test_loss,test_acc,test_iou,pseudo_acc_list,model1_loss ):
  save_path = "./pseudo_exp/" + str(image1Folder) + "ratio" + str(pseudo_ratio) + "/"
  file_name = "set" + str(num_folder_label) + ".xlsx"
  workbook = xlsxwriter.Workbook(save_path + file_name)
  worksheet = workbook.add_worksheet()

  row_numEpoc = 0
  row_time = 1
  row_epochLoss = 2
  row_validLoss = 3
  row_iou = 4
  row_acc = 5
  row_test_loss = 6
  row_testiou = 7
  row_testacc = 8
  row_model1_loss = 9
  row_model1_iou = 10
  row_pseudo_acc = 11


  worksheet.write(0, row_numEpoc, "epoch")
  worksheet.write(1, row_numEpoc, epoch_label)
  worksheet.write(0, row_time, "time")
  worksheet.write(1, row_time, time_used)
  worksheet.write(0, row_epochLoss , "train loss")
  worksheet.write(0, row_validLoss , "valid loss")
  worksheet.write(0, row_iou , "valid IOU")
  worksheet.write(0, row_acc , "valid acc")
  worksheet.write(0, row_test_loss , "test loss")
  worksheet.write(0, row_testiou , "test IOU")
  worksheet.write(0, row_testacc , "test acc")
  worksheet.write(0, row_model1_loss , "model1 loss")
  worksheet.write(0, row_model1_iou , "model1 iou")
  worksheet.write(0, row_pseudo_acc , "pseudo-acc")



  test_loss = str(test_loss).split(".")
  if len(test_loss[1])>5:
      test_loss[1] = test_loss[1][0:5]
  test_loss =  float((".").join(test_loss))

  test_iou = str(test_iou).split(".")
  if len(test_iou[1])>5:
      test_iou[1] = test_iou[1][0:5]
  test_iou =  float((".").join(test_iou))

  test_acc = str(test_acc).split(".")
  if len(test_acc[1])>5:
      test_acc[1] = test_acc[1][0:5]
  test_acc =  float((".").join(test_acc))

  model1_loss = str(model1_loss).split(".")
  if len(model1_loss[1])>5:
      model1_loss[1] = model1_loss[1][0:5]
  model1_loss =  float((".").join(model1_loss))

  model1_iou = str(model1_iou).split(".")
  if len(model1_iou[1])>5:
      model1_iou[1] = model1_iou[1][0:5]
  model1_iou =  float((".").join(model1_iou))
  ''' pseudo_acc change to a list as updateing pseudos
  pseudo_acc = str(pseudo_acc).split(".")
  if len(pseudo_acc[1])>5:
      pseudo_acc[1] = pseudo_acc[1][0:5]
  pseudo_acc =  float((".").join(pseudo_acc))
  '''


  worksheet.write(1, row_test_loss , test_loss)
  worksheet.write(1, row_testiou , test_iou)
  worksheet.write(1, row_testacc , test_acc)
  worksheet.write(1, row_model1_loss , model1_loss)
  #worksheet.write(1, row_pseudo_acc , pseudo_acc)


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
  for i in range(len(pseudo_acc_list)):
    pa = str(pseudo_acc_list[i]).split(".")
    if len(pa[1])>5:
          pa[1] = pa[1][0:5]
    pseudo_acc_list[i] = float((".").join(pa))


  worksheet.write_column(1,row_epochLoss,epoch_loss_list)
  worksheet.write_column(1,row_validLoss,epoch_valid_loss_list)
  worksheet.write_column(1,row_iou,epoch_valid_iou_list)
  worksheet.write_column(1,row_acc,epoch_valid_acc_list)
  worksheet.write_column(1,row_pseudo_acc,pseudo_acc_list)
  workbook.close()
  print('Finished writing result')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def validate(model):
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

    return epoch_valid_loss, epoch_valid_acc,epoch_valid_iou


def lineDrawingPred(predicted_labels):
  # input dimension (h,w)
  h, w = predicted_labels.shape
  #print(h,w)
  idx_h = []
  idx_w = []
  for i in range(1,h-2):
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
  idx_h = []
  idx_w = []
  for i in range(1,h-2):
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

def plotAU(AUloader):
    trans1 = transforms.ToPILImage()

    for i, batch in enumerate(AUloader):
        result = Image.new("RGBA",(1024*2, 512))
        
        image = batch[0][0]
        target_graph = batch[1][0]*10
        #print(torch.unique(target_graph))
        trans1 = transforms.ToPILImage()

        image = trans1(image).convert("RGBA")
        target_graph = trans1(target_graph).convert("RGBA")

        result.paste(image, box = (0,0))
        result.paste(target_graph, box = (1024,0))

        save_name = "./pseudo_exp/5ratio" + str(pseudo_ratio) + "/AU" + str(i) + ".tif"
        result.save(save_name,"TIFF")
        print("save"+ str(i))


label_path = "./labelled_OCT/"
original_path = "./original_OCT/"
labelled_image_file = os.listdir(label_path)
original_image_file = os.listdir(original_path)

'''
for i in range(20):
    label_folder_path = os.path.join(label_path + labelled_image_file[i])

    original_folder_path = os.path.join(original_path + labelled_image_file[i])
    print(label_folder_path)
    print(original_folder_path)
'''

for i in range(num_folder_train, num_folder_train+num_folder_valid):
    label_folder_path = os.path.join(label_path + labelled_image_file[i])

    original_folder_path = os.path.join(original_path + labelled_image_file[i])


    if i == num_folder_train:
      validSet = OCT_BScan(original_folder_path,label_folder_path)
    else:
      validSet += OCT_BScan(original_folder_path,label_folder_path)
validLoader = DataLoader(validSet,batch_size=1)
print("total images for validation:",len(validSet))



for i in range( num_folder_train+num_folder_valid, len(labelled_image_file )):
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + labelled_image_file[i])

    if i ==  num_folder_train+num_folder_valid:
        testSet = OCT_BScan(original_folder_path,label_folder_path)
    else:
        testSet += OCT_BScan(original_folder_path,label_folder_path)
testLoader = DataLoader(testSet,batch_size=1)
print("total images for test :",len(testSet))


def test4DataLoader():
  for i, batch in enumerate(labelLoader):
    if i == 0:
      input_graph = batch[0]
      target_graph = batch[1]
      print(input_graph.shape)
      print(target_graph.shape)
      target_stack = target_graph
      break
#test4DataLoader()

#Training Process:
print("start trainning")
triggered = 0

# Trainning Loop
while num_folder_label < folder_label_range:
    model_label = UnetModel(1,1).to(device)
    #model_final = UnetModel(1,1).to(device)
    optimizer_label = optim.Adam(model_label.parameters(), lr=1e-5)
    #scheduler_label = optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, "max", patience = 2)
    scheduler_label = optim.lr_scheduler.StepLR(optimizer_label, step_size=80, gamma=0.1)
    #optimizer_final = optim.Adam(model_final.parameters(), lr=0.0001)

    epoch_loss_list = []
    epoch_valid_acc_list = []
    epoch_valid_iou_list = []
    epoch_valid_loss_list = []
    pseudo_acc_list = []
    logpath = "./log/"
    writer = SummaryWriter(logpath)
    start_time = time()
    best_valid_acc = 0
    best_valid_iou = 0
    best_valid_loss = 100
    test_acc = 0
    test_iou = 0
    test_loss = 0
    model1_loss = 0
    model1_iou = 0
    pseudo_acc = 0
    for i in range(num_folder_label):
        label_folder_path = os.path.join(label_path + labelled_image_file[i])
        original_folder_path = os.path.join(original_path + labelled_image_file[i])
        len_folder = len(os.listdir(label_folder_path))
        scans = OCT_BScan(original_folder_path,label_folder_path)
        pseudo_num = image1Folder*pseudo_ratio
        else_num = len_folder - pseudo_num
        if i == 0 :
            labelSet =   torch.utils.data.Subset(scans, range(4,100,int(100/image1Folder)))
            #pseudoSet = torch.utils.data.Subset(scans, range(image1Folder,image1Folder*(pseudo_ratio+1)))
            pseudoSet, _ = torch.utils.data.random_split(scans, [pseudo_num,else_num])
        else:
            labelSet  += torch.utils.data.Subset(scans, range(4,100,int(100/image1Folder)))
            #pseudoSet += torch.utils.data.Subset(scans, range(image1Folder,image1Folder*(pseudo_ratio+1)))
            pSet, _ = torch.utils.data.random_split(scans, [pseudo_num,else_num])
            pseudoSet += pSet
        for j in range(num_augmentation):
            scans_aug = OCT_BScan_AU(original_folder_path,label_folder_path)
            labelSet  += torch.utils.data.Subset(scans_aug, range(4,100,int(100/image1Folder)))
            
            # pseudo_augmentation
            pSet, _ = torch.utils.data.random_split(scans_aug, [pseudo_num,else_num])
            pseudoSet += pSet
            

    print("total images for labelled training:",len(labelSet))
    labelLoader= DataLoader(labelSet,batch_size=1)
    '''
    for i in range(num_folder_label, num_folder_train):
        label_folder_path = os.path.join(label_path + labelled_image_file[i])
        original_folder_path = os.path.join(original_path + labelled_image_file[i])
        scans = OCT_BScan(original_folder_path,label_folder_path)
        #pseudoSet += torch.utils.data.Subset(scans, range(0,image1Folder*pseudo_ratio))
        pseudoSet, _ += torch.utils.data.random_split(scans, [pseudo_num,else_num])
    '''
    pseudoLoader = DataLoader(pseudoSet,batch_size=1)
    
    print("total images for pseudolabel:",len(pseudoSet))

    # model first trainning
    for epoch in range(num_epoch):

        running_loss = 0
        for i, batch in enumerate(labelLoader):
            input_graph = batch[0].to(device)
            target_graph = batch[1]
            target_label = graph2mask(target_graph).to(device)

            optimizer_label.zero_grad()
            output_label = model_label(input_graph).to(device)

            loss = loss_function(output_label.to(torch.float32),target_label.to(torch.float32))
            loss.backward()
            optimizer_label.step()

            running_loss += loss.item()
        model1_loss = running_loss/len(labelSet)
        epoch_valid_loss, epoch_valid_acc, epoch_valid_iou = validate(model_label)

        epoch_loss_list.append(model1_loss)
        epoch_valid_loss_list.append(epoch_valid_loss)
        epoch_valid_acc_list.append(epoch_valid_acc)
        epoch_valid_iou_list.append(epoch_valid_iou)

        #scheduler_label.step(epoch_valid_iou)
        scheduler_label.step()
        print("lr: ", get_lr(optimizer_label))
        model1_iou = epoch_valid_iou
        print(model1_iou)
        epoch_label = epoch + 1
        if (epoch_valid_iou > 0.95):
            break


    print("model first training finished", model1_loss)
    result = Image.new("RGBA",(1024, 4*512))
    sv_image = int(len(pseudoSet)/4)
    col = 0
    # prediction process
    for i, batch in enumerate(pseudoLoader):
        input_img = batch[0].to(device)
        output_label = binaryMask(model_label(input_img).cpu())
        pseudolabels[i] = output_label
        target = graph2mask(batch[1])
        pseudo_acc += float(acc_fcn(output_label,target))


        if ((i+1)%sv_image == 0):
            pred = mask2graph(output_label).view(512,1024)
            label = mask2graph(target.view(512,1024))
            predLines = lineDrawingPred(pred)
            labelLines = lineDrawingLabel(label)
            background = input_img.view(512,1024).cpu()

            trans1 = transforms.ToPILImage()

            background = trans1(background).convert("RGBA")
            overlayPred = trans1(predLines).convert("RGBA")
            overlayLabel = trans1(labelLines).convert("RGBA")
            overlay = Image.blend(overlayPred, overlayLabel, 0.5)
            new_img = Image.blend(background, overlay, 0.6)


            result.paste(new_img, box = (0, col*512))
            col += 1
    save_file = "./pseudo_exp/" + str(image1Folder) + "ratio" + str(pseudo_ratio) + "/"
    save_name =  save_file + "pseudo" + str(num_folder_label) + "_E0.tif"
    result.save(save_name,"TIFF")
    if len(pseudoSet) != 0:
        pseudo_acc /= len(pseudoSet)
    else:
        pseudo_acc = 0.0
    
    pseudo_acc_list.append(pseudo_acc)
    print("model1 trained and labels predicted, start training with pseudo-labels")
  
    

    # Training with  pseudo-labels
    for epoch in range(num_epoch):

        running_loss = 0
        for i, batch in enumerate(pseudoLoader):
            input_graph = batch[0].to(device)

            target_graph = pseudolabels[i].to(device)

            optimizer_label.zero_grad()
            output_label = model_label(input_graph).numpy.to(device)

            loss = 3*loss_function(output_label.to(torch.float32),target_label.to(torch.float32))
            loss.backward()
            optimizer_label.step()
            running_loss += loss.item()

        for i, batch in enumerate(labelLoader):
            input_graph = batch[0].to(device)
            target_graph = batch[1]
            target_label = graph2mask(target_graph).to(device)

            optimizer_label.zero_grad()
            output_label = model_label(input_graph).to(device)

            loss = loss_function(output_label.to(torch.float32),target_label.to(torch.float32))
            loss.backward()
            optimizer_label.step()

            running_loss += loss.item()
        epoch_loss = running_loss/(len(pseudoSet)+len(labelSet))


        epoch_loss_list.append(epoch_loss)
        # validation:
        epoch_valid_loss, epoch_valid_acc, epoch_valid_iou = validate(model_label)

        scheduler_label.step()
        print("lr: ", get_lr(optimizer_label))
        epoch_valid_loss_list.append(epoch_valid_loss)
        epoch_valid_acc_list.append(epoch_valid_acc)
        epoch_valid_iou_list.append(epoch_valid_iou)

        writer.add_scalar('train loss/epoch', epoch_loss, epoch)
        writer.add_scalar('valid loss/epoch', epoch_valid_loss, epoch)
        writer.add_scalar('acc/epoch', epoch_valid_acc, epoch)
        writer.add_scalar('iou/epoch', epoch_valid_iou, epoch)
        print(epoch, "loss, acc, iou",
                epoch_valid_loss, epoch_valid_acc, epoch_valid_iou)
        if (earlyStoppingFlag):
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                triggered = 0
            else:
                if abs(epoch_loss_list[-1]-epoch_loss_list[-2]) < 10**(-6):
                    break
                triggered += 1

                print("triggered")

                if (triggered > patience):
                        print("early stopping")
                        break

        # prediction process
        pseudo_acc = 0
        col = 0
        image_pseudo = Image.new("RGBA",(1024, 4*512))
        pseudolabels = torch.zeros(len(pseudoSet),1,1,512,1024)
        for i, batch in enumerate(pseudoLoader):
            input_img = batch[0].to(device)
            output_label = binaryMask(model_label(input_img).cpu())
            pseudolabels[i] = output_label
            target = graph2mask(batch[1])
            pseudo_acc += float(acc_fcn(output_label,target))


            if ((i+1)%sv_image == 0) and ((epoch+1)%10 == 0):
                print("pseudo_image generating")
                pred = mask2graph(output_label).view(512,1024)
                label = mask2graph(target.view(512,1024))
                predLines = lineDrawingPred(pred)
                labelLines = lineDrawingLabel(label)
                background = input_img.view(512,1024).cpu()


                trans1 = transforms.ToPILImage()

                background = trans1(background).convert("RGBA")
                
                overlayPred = trans1(predLines).convert("RGBA")
                overlayLabel = trans1(labelLines).convert("RGBA")
                overlay = Image.blend(overlayPred, overlayLabel, 0.5)
                new_img = Image.blend(background, overlay, 0.6)
                #new_img.save("./try.tif","TIFF")


                image_pseudo.paste(new_img, box = (0, col*512))
                col += 1
        if (epoch+1)%10 == 0:
            save_file = "./pseudo_exp/" + str(image1Folder) + "ratio" + str(pseudo_ratio) + "/"
            save_name =  save_file + "pseudo" + str(num_folder_label) + "_E"+ str(epoch+1) + ".tif"
            image_pseudo.save(save_name,"TIFF")
        if len(pseudoSet) != 0:
            pseudo_acc /= len(pseudoSet)
        else:
            pseudo_acc = 0.0
        pseudo_acc_list.append(pseudo_acc)



    print("Training done")

    end_time = time()
    seconds_elapsed = end_time - start_time
    mins, rest = divmod(seconds_elapsed, 60)
    mins = str(mins).split(".")
    rest = str(rest).split(".")
    time_used =  mins[0] + "." + rest[0]

    for i, batch in enumerate(testLoader):
        input_graph = batch[0].to(device)
        label = graph2mask(batch[1])
        pred_label = model_label(input_graph).cpu()
        loss = loss_function(pred_label.to(torch.float32),
                                label.to(torch.float32))
        test_loss += loss.item()
        pred = binaryMask(pred_label)
        test_iou += float(iou_fcn(pred,label)[0])
        test_acc += float(acc_fcn(pred,label))
    test_acc /= len(testSet)
    test_iou /= len(testSet)
    test_loss /= len(testSet)


    #Writing Result to Excel
    print('Test done and writing result')
    write_result(time_used,epoch_loss_list,model1_iou,
                epoch_valid_acc_list,epoch_valid_iou_list,epoch_valid_loss_list,
                test_loss,test_acc,test_iou,pseudo_acc_list,model1_loss)

    save_path = "./pseudo_exp/" + str(image1Folder) + "ratio" + str(pseudo_ratio) + "/"
    model_name = "set" + str(num_folder_label) + ".pt"
    torch.save(model_label.state_dict(), save_path + model_name)
    print("model saved")
    num_folder_label += 1
