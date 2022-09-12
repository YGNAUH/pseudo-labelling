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

import tensorrt as trt
import time
import pycuda.driver as cuda
import pycuda.autoinit
import scipy.spatial




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = UnetModel(1,1).to(device)
model_file = "./pseudo_more/5ratio1/set5.pt"
model.load_state_dict(torch.load(model_file))
ONNX_FILE_PATH = './pseudo_more/5ratio1/set5.onnx'
engine_file_path = './pseudo_more/5ratio1/set5.engine'
#engine_file_path = './pseudo_exp/pse_AU/set5_FP16.engine'

def saveONNX():
    dummy_input = torch.randn(1, 1, 512, 1024, device='cuda')
    torch.onnx.export(model, dummy_input, ONNX_FILE_PATH, verbose=True)

def ONNX_build_engine(onnx_file_path, engine_file_path):
    '''
    loading onnx file and build engine
    :param onnx_file_path: onnx file path
    :param engine_file_path: engine file path
    :return: engine
    '''

 
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    batch_size = 1 
    
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:
    
        builder.max_batch_size = batch_size
    
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  
        config.set_flag(trt.BuilderFlag.FP16)
        
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
    
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(
            onnx_file_path))
    
        # Dynanic input
        '''
        profile = builder.create_optimization_profile()
        profile.set_shape("input_1", (1, 32, 128, 3),
                        (1, 32, 128,  3), (1, 32, 128, 3))
        config.add_optimization_profile(profile)
        '''
    
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
    
        # save file
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        
        return engine


# SAVE ONNEX FILDE AND ENGINE FILE
saveONNX()
ONNX_build_engine(ONNX_FILE_PATH,engine_file_path)




# test loader
num_folder_train = 14
num_folder_valid = 3
idx_folder_test = num_folder_train + num_folder_valid

label_path = "./labelled_OCT/"
original_path = "./original_OCT/"
labelled_image_file = os.listdir(label_path)
original_image_file = os.listdir(original_path)


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


for i in range(idx_folder_test, len(labelled_image_file )):
    print(labelled_image_file[i])
    label_folder_path = os.path.join(label_path + labelled_image_file[i])
    original_folder_path = os.path.join(original_path + labelled_image_file[i])
    testSet = OCT_BScan_test(original_folder_path,label_folder_path)
    if i == idx_folder_test:
        test_images = testSet
    else:
        test_images += testSet
testLoader = DataLoader(test_images,batch_size=1)
print(len(test_images))


def loadEngine2TensorRT(filepath):
    #G_LOGGER = trt.Logger(trt.Logger.WARNING)
    G_LOGGER = trt.Logger()
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine


def acc_fcn(outputs, labels):
    correct = (outputs == labels).float()
    acc = correct.sum()/correct.numel()
    return acc

def binaryMask(labels):
  graph_output = torch.where((labels > 0.5), torch.tensor(1),
        torch.tensor(0))
  return graph_output

def mask2graph(labels):
  graph_output = torch.where(labels > 0.5, torch.tensor(0.0392),
   torch.tensor(0.0))
  return graph_output


def graph2mask(graph_input):
  label_output = torch.where((graph_input > 0.00) & (graph_input < 0.0392),
        torch.tensor(1), torch.tensor(0))
  return label_output



def inference_hd():
    HD_trt = 0
    HD_pt = 0

    time_trt = 0
    time_pt = 0

    for i, batch in enumerate(testLoader):
        img = batch[0]
        target = batch[1]
    
        #img, target = next(iter(testLoader))
        hd_p, time_p = modelComparison_hd(img,target)

        HD_pt += hd_p
        time_pt += time_p

        img = img.numpy()
        target = target.numpy()

        img = img.ravel()
        target = target.ravel()


        engine = loadEngine2TensorRT(engine_file_path)
        context = engine.create_execution_context()
        output = np.empty((1,1,512,1024), dtype=np.float32)

        #allocate mem
        d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
        bindings = [int(d_input), int(d_output)]

        batch_size = 1
        #pycuda stream
        stream = cuda.Stream()

        #input data to device
        cuda.memcpy_htod_async(d_input, img, stream)

        ### timer ###
        start = time.time()
        #execute model
        context.execute_async(batch_size, bindings, stream.handle, None)
        ### timer ###
        end = time.time()

        #output from the stream
        cuda.memcpy_dtoh_async(output, d_output, stream)
        
        
        #synchronize
        stream.synchronize()

    
        output = torch.from_numpy(output).view(512,1024)
        target = torch.from_numpy(target).view(512,1024)

        output = mask2graph(binaryMask(output))

        hd = scipy.spatial.distance.directed_hausdorff(output, target)
        #print("tensorRT HD metric: ",hd[0])
        HD_trt += hd[0]

        time_trt += end - start
        #print("tensorrt time:", end - start)

        del context
        del engine

    HD_trt /= len(test_images)
    HD_pt /= len(test_images)
    time_trt /= len(test_images)
    time_pt /= len(test_images)
    print("TensorRT HD, TIME:", HD_trt, time_trt)
    print("PyTorchch HD, TIME:", HD_pt, time_pt)

def modelComparison_hd(img, target):
    #print("Pytorch Running")
    model = UnetModel(1,1).to(device)
    model_file = "./pseudo_exp/pse_AU/set5.pt"
    model.load_state_dict(torch.load(model_file))


    image = img.to(device)

    target = target.view(512,1024)
    start = time.time()
    pred_label = model(image).view(512,1024).cpu()
    pred = mask2graph(binaryMask(pred_label))
    end = time.time()
    hd = scipy.spatial.distance.directed_hausdorff(pred, target)
    #print("PyTorch HD metric: ",hd[0])
    #print("PyTorch Time:", end-start)
    
    t = end - start

    return hd[0],t
#inference_hd()


def inference_acc():
    acc_trt = 0
    acc_pt = 0

    time_trt = 0
    time_pt = 0

    for i, batch in enumerate(testLoader):
        img = batch[0]
        target = batch[1]
    
        #img, target = next(iter(testLoader))
        acc_p, time_p = modelComparison_acc(img,target)

        acc_pt += acc_p
        time_pt += time_p

        img = img.numpy()
        target = target.numpy()

        img = img.ravel()
        target = target.ravel()


        engine = loadEngine2TensorRT(engine_file_path)
        context = engine.create_execution_context()
        output = np.empty((1,1,512,1024), dtype=np.float32)

        #allocate mem
        d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
        bindings = [int(d_input), int(d_output)]

        batch_size = 1
        #pycuda stream
        stream = cuda.Stream()

        #input data to device
        cuda.memcpy_htod_async(d_input, img, stream)

        ### timer ###
        start = time.time()
        #execute model
        context.execute_async(batch_size, bindings, stream.handle, None)
        ### timer ###
        end = time.time()

        #output from the stream
        cuda.memcpy_dtoh_async(output, d_output, stream)
        
        
        #synchronize
        stream.synchronize()

    
        output = torch.from_numpy(output).view(512,1024)
        target = torch.from_numpy(target).view(512,1024)
        label = graph2mask(target)
        pred = binaryMask(output)
        acc_trt += float(acc_fcn(pred,label))


        time_trt += end - start
        #print("tensorrt time:", end - start)

        del context
        del engine

    acc_trt /= len(test_images)
    acc_pt /= len(test_images)
    time_trt /= len(test_images)
    time_pt /= len(test_images)
    print("TensorRT,ACC, TIME:", acc_trt, time_trt)
    print("PyTorchch aCC TIME:", acc_pt, time_pt)

def modelComparison_acc(img, target):
    #print("Pytorch Running")
    model = UnetModel(1,1).to(device)
    model_file = "./pseudo_exp/pse_AU/set5.pt"
    model.load_state_dict(torch.load(model_file))


    image = img.to(device)

    target = target.view(512,1024)
    start = time.time()
    pred_label = model(image).view(512,1024).cpu()
    end = time.time()
    
    label = graph2mask(target)
    pred = binaryMask(pred_label)
    acc = float(acc_fcn(pred,label))
    
    
    hd = scipy.spatial.distance.directed_hausdorff(pred, target)
    #print("PyTorch HD metric: ",hd[0])
    #print("PyTorch Time:", end-start)
    
    t = end - start

    return acc,t
#inference_acc()


def iou_fcn(outputs, labels):
    # BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    #labels = labels.squeeze(1)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection) / (union)  # We smooth our devision to avoid 0/0

    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou



#def inference_iou():



def cal_engine_iou():
    time_trt = 0
    iou_trt = 0
    count = 0
    for i, batch in enumerate(testLoader):
        img = batch[0]
        target = batch[1]

        img = img.numpy()
        target = target.numpy()

        img = img.ravel()
        target = target.ravel()


        engine = loadEngine2TensorRT(engine_file_path)
        context = engine.create_execution_context()
        output = np.empty((1,1,512,1024), dtype=np.float32)

        #allocate mem
        d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
        bindings = [int(d_input), int(d_output)]

        batch_size = 1
        #pycuda stream
        stream = cuda.Stream()

        #input data to device
        cuda.memcpy_htod_async(d_input, img, stream)

        ### timer ###
        start = time.time()
        #execute model
        context.execute_async(batch_size, bindings, stream.handle, None)
        ### timer ###
        end = time.time()

        #output from the stream
        cuda.memcpy_dtoh_async(output, d_output, stream)
        
        
        #synchronize
        stream.synchronize()

    
        output = torch.from_numpy(output).view(1,512,1024)
        target = torch.from_numpy(target).view(1,512,1024)
        label = graph2mask(target)
        pred = binaryMask(output)
        print(pred.shape, label.shape)
        print(torch.unique(pred), torch.unique(label))
        cur_iout = float(iou_fcn(pred,label))

        iou_trt += cur_iout
        count += 1
        print(cur_iout)
        print(iou_trt)


        time_trt += end - start
        print("tensorrt time:", end - start)

        del context
        del engine
        if i == 298:
            break

    iou_trt /= count
    time_trt /= count

    return iou_trt, time_trt



def inference_pt():
    model = UnetModel(1,1).to(device)
    model_file = "./pseudo_more/5ratio1/set5.pt"
    model.load_state_dict(torch.load(model_file))
    time_pt = 0
    for i, batch in enumerate(testLoader):
        img = batch[0]
        target = batch[1]
        image = img.to(device)

        target = target.view(1,512,1024)
        start = time.time()
        pred_label = model(image).view(1, 512,1024).cpu()
        end = time.time()
        time_pt += end-start
    time_pt /= len(testSet)
    print("pt time", time_pt)
    return time_pt
iou_trt, time_trt = cal_engine_iou()
print("the iou for engine:", iou_trt)
print("the time for engie:", time_trt)
time_pt = inference_pt()
print("the time for pt", time_pt)