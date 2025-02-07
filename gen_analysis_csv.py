import pandas as pd
from pydicom import dcmread
import models
import torch
from accessory_functions import dicom_path_to_tensor
import argparse

#read in test dataset
#extract z-axis, time, PID
#calculate model scores
#save score dfs

parser = argparse.ArgumentParser(description='Generate CSVs to further analyze data')
parser.add_argument('--data_path', type=str, required=True, help='path to test set csv') #will need to modify this part to get proper test set examples
parser.add_argument('--model', type=str,choices=['vgg11','resnet18','resnet34','resnet50'],required=True,help='Which model architecture to use')
parser.add_argument('--model_weights', type=str,required=True,help='Path to trained model weights')
parser.add_argument('--save_path', type=str,help='path to save dataset to',required=True)

args = parser.parse_args()

dataset_path = args.data_path
df = pd.read_csv(dataset_path,header=None)
if args.model == 'resnet18':
    model = models.resnet18(num_channels=5)
elif args.model == 'resnet34':
    model = models.resnet34(num_channels=5)
elif args.model == 'resnet50':
    model = models.resnet50(num_channels=5)
elif args.model == 'vgg11':
    model = models.vgg11(num_channels=5)
model.load_state_dict(torch.load(args.model_weights))
model.eval()
all_z_axis = []
all_time = []
all_pid = []
all_scores = []
print(f'Length of dataset: {len(df)}')

for i,path in enumerate(df[0]):
    dcm = dcmread(path)
    pid = dcm.PatientID
    time = dcm.TriggerTime
    loc = dcm.get((0x0020,0x1041),'Unknown').value
    img = dicom_path_to_tensor(path).unsqueeze(0)
    score = model(img)
    score_sigmoid = torch.sigmoid(score)
    score_sigmoid = score_sigmoid[0,0].item()
    all_z_axis.append(loc)
    all_time.append(time)
    all_pid.append(pid)
    all_scores.append(score_sigmoid)
    if i % 1000 == 0:
        print(i)

df['z_coord'] = all_z_axis
df['pid'] = all_pid
df['time'] = all_time
df['pred'] = all_scores
df.rename(columns={0:'dcm_path',1:'label'},inplace=True)
df.to_csv(args.save_path,index=False)