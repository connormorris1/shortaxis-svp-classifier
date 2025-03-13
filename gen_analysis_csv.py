import pandas as pd
from pydicom import dcmread
import models
import torch
from accessory_functions import dicom_path_to_tensor
import random
import os
import argparse

#read in test dataset
#extract z-axis, time, PID
#calculate model scores
#save score dfs

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
model.load_state_dict(torch.load(args.model_weights))
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