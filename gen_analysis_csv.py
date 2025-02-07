import pandas as pd
from pydicom import dcmread
import models
import torch
from accessory_functions import CustomImageDataset, dicom_path_to_tensor3d
import argparse
import random
import os

#read in test dataset
#extract z-axis, time, PID
#calculate model scores
#save score dfs

parser = argparse.ArgumentParser(description='Generate CSVs to further analyze data')
parser.add_argument('--base_path', type=str, required=True, help='path to parent directory of dataset')
parser.add_argument('--model', type=str,choices=['vgg11','resnet18','resnet34','resnet50'],required=True,help='Which model architecture to use')
parser.add_argument('--model_weights', type=str,required=True,help='Path to trained model weights')
parser.add_argument('--save_path', type=str,help='path to save dataset to',required=True)
args = parser.parse_args()

base_path = args.base_path
print(f"Generating analysis CSV for {base_path} with model {args.model} and weights {args.model_weights}.")
if args.model == 'resnet18':
    model = models.resnet18(num_channels=2)
elif args.model == 'resnet34':
    model = models.resnet34(num_channels=2)
elif args.model == 'resnet50':
    model = models.resnet50(num_channels=2)
elif args.model == 'vgg11':
    model = models.vgg11(num_channels=2)
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

def gen_analysis_df(pid_paths,labels,df):
    #loop through all test images, extract metadata, test on model
    dataset = CustomImageDataset(pid_paths, labels,train=False)
    for i,img_set in enumerate(dataset.all_paths):
        img,_ = dataset[i]
        img = img.unsqueeze(0)
        score = model(img)
        score_sigmoid = torch.sigmoid(score)
        score_sigmoid = score_sigmoid[0,0].item()
        all_locs = []
        all_times = []
        for j,img_path in enumerate(img_set):
            dcm = dcmread(img_path)
            time = float(dcm.TriggerTime)
            all_times.append(time)
            loc = float(dcm.get((0x0020,0x1041),'Unknown').value) 
            all_locs.append(loc)
            if j == 0:
                pid = dcm.PatientID
                date = dcm.StudyDate
                series_desc = dcm.SeriesDescription
        df.loc[len(df)] = [img_set,pid,date,series_desc,all_locs,all_times,score_sigmoid,dataset.all_labels[i]] 
    return df
print("Generating analysis dataframe")
df = gen_analysis_df(test_svp_pids + test_normal_pids,[1]*len(test_svp_pids) + [0]*len(test_normal_pids),df)
print(f"Saving analysis dataframe to {args.save_path}")
df.to_csv(args.save_path,index=False)