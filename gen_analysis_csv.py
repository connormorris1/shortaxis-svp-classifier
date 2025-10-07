import pandas as pd
from pydicom import dcmread
import models
import torch
# from accessory_functions import dicom_path_to_tensor
import random
import os
import argparse
from torchvision import transforms
import numpy as np

#read in test dataset
#extract z-axis, time, PID
#calculate model scores
#save score dfs

def dicom_path_to_tensor(img_path,target_dim,train=True):
    # Load image dicom and pre-process into tensor for input into model
    dicom = dcmread(img_path)
    array = dicom.pixel_array
    #rescale to 1mmx1mm pixel size - not working currently
    # array = rescale(array, dicom.PixelSpacing,anti_aliasing=dicom.PixelSpacing[0] < 1)
    #center crop to target_dim x target_dim, with 0 padding if less
    array = array.astype(np.uint8)
    if train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1181772], std=[0.13326852023601532]),
            transforms.Pad((int(np.ceil(max(target_dim-array.shape[0],0)/2)),int(np.ceil(max(target_dim-array.shape[1],0)/2)))), #can add rotation/flipping here
            transforms.CenterCrop((target_dim,target_dim)),
            transforms.RandomRotation(degrees=(0,180)), #transforms copied from https://www.sciencedirect.com/science/article/pii/S1097664723003320?via%3Dihub, but instead of random transforms they did all permutations on each image
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
    else: # remove data augmentation when testing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1181772], std=[0.13326852023601532]),
            transforms.Pad((int(np.ceil(max(target_dim-array.shape[0],0)/2)),int(np.ceil(max(target_dim-array.shape[1],0)/2)))), #can add rotation/flipping here
            transforms.CenterCrop((target_dim,target_dim)),
        ])
    array = transform(array)
    return array

parser = argparse.ArgumentParser(description='Generate CSVs to further analyze data')
parser.add_argument('--base_path', type=str, required=True, help='path to parent directory of dataset')
parser.add_argument('--model', type=str,choices=['vgg11','resnet18','resnet34','resnet50'],required=True,help='Which model architecture to use')
parser.add_argument('--model_weights', type=str,required=True,help='Path to trained model weights')
parser.add_argument('--save_path', type=str,help='path to save dataset to',required=True)
parser.add_argument('--all',action='store_true',help='Use all PIDs for analysis, rather than only test set')

args = parser.parse_args()

base_path = args.base_path
print(f"Generating analysis CSV for {base_path} with model {args.model} and weights {args.model_weights}.")
if args.model == 'resnet18':
    model = models.resnet18()
elif args.model == 'resnet34':
    model = models.resnet34()
elif args.model == 'resnet50':
    model = models.resnet50()
elif args.model == 'vgg11':
    model = models.vgg11()
model.load_state_dict(torch.load(args.model_weights,map_location=torch.device('cpu')))
model.eval()

#extract test set PID paths
normal_path = os.path.join(base_path,'normal')
normal_pids = [] #will be list of full paths to all pid directories
for pid in os.listdir(normal_path):
    normal_pids.append(os.path.join(normal_path,pid))

svp_path = os.path.join(base_path,'svp')
svp_pids = [] #will be list of full paths to all pid directories
for pid in os.listdir(svp_path):
    svp_pids.append(os.path.join(svp_path,pid))

random.seed(10)
test_svp_pids = random.sample(svp_pids,6) #each PID should be a full path to that PID directory
test_normal_pids = random.sample(normal_pids,9)

all_z_axis = []
all_time = []
all_pid = []
all_scores = []
df = pd.DataFrame(columns=['path','pid','date','series_description','z_coord','time','pred','label'])

def gen_analysis_df(pid_paths,label,df):
    #loop through all test images, extract metadata, test on model
    for pid_path in pid_paths:
        date_paths = os.listdir(pid_path)
        for date_path in date_paths:
            pid_date_path = os.path.join(pid_path,date_path)
            for sequence in os.listdir(pid_date_path):
                seq_path = os.path.join(pid_date_path,sequence)
                for dcm in os.listdir(seq_path):
                    dcm_path = os.path.join(seq_path,dcm)
                    dcm = dcmread(dcm_path)
                    pid = dcm.PatientID
                    time = float(dcm.TriggerTime)
                    loc = float(dcm.get((0x0020,0x1041),'Unknown').value)
                    img = dicom_path_to_tensor(dcm_path,224,train=False).unsqueeze(0)
                    date = dcm.StudyDate
                    series_desc = dcm.SeriesDescription
                    score = model(img)
                    score_sigmoid = torch.sigmoid(score)
                    score_sigmoid = score_sigmoid[0,0].item()
                    df.loc[len(df)] = [dcm_path,pid,date,series_desc,loc,time,score_sigmoid,label] 
    return df
print("Generating analysis dataframe for SVPs")
if args.all:
    print("Using all PIDs for analysis")
    df = gen_analysis_df(svp_pids,1,df)
else:
    print("Using only test PIDs for analysis")
    df = gen_analysis_df(test_svp_pids,1,df)
print("Generating analysis dataframe for normals")
if args.all:
    df = gen_analysis_df(normal_pids,0,df)
else:
    df = gen_analysis_df(test_normal_pids,0,df)
print(f"Saving analysis dataframe to {args.save_path}")
df.to_csv(args.save_path,index=False)