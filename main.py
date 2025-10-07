import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import time
import wandb
import argparse
from accessory_functions import CustomImageDataset, train_loop, test_loop
from models import resnet18, resnet34, resnet50, vgg11
import random
import os

############################################################################
parser = argparse.ArgumentParser(description='Begin training runs')
parser.add_argument('--normal_path',type=str,required=True,help='path to normal examples')
parser.add_argument('--svp_path', type=str,required=True,help='path to svp examples')
parser.add_argument('--fold', type=int,required=True,choices=[0,1,2,3],help='Fold number (0-3)')
parser.add_argument('--resolution', type=int,default=224,help='resolution to resize images to')
parser.add_argument('--batch_size', type=int,default=50,help='batch size')
parser.add_argument('--epochs', type=int,default=3,help='number of epochs')
parser.add_argument('--save_model_path', type=str,help='path to save model to, should end in .pth',required=True)
parser.add_argument('--run_name', type=str,help='Run name in WandB',required=True)
parser.add_argument('--wandb_key',type=str,help='Path to file containing WandB login key.',required=True)
parser.add_argument('--pretrained', action='store_true',help='Use pretrained resnet model')
parser.add_argument('--architecture', type=str,choices=['resnet18','resnet34','resnet50','vgg11'],default='resnet18',help='Which architecture to use')
parser.add_argument('--device',type=int,required=True,help='Cuda GPU ID')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='Sets learning rate')
parser.add_argument('--dataset_balancing',type=str,choices=['loss_weighting','oversampling'],default='loss_weighting',help='Choose method for correcting dataset imbalance.')
parser.add_argument('--lr_scheduler',action='store_true',help='Use learning rate scheduler')
parser.add_argument('--freeze_encoder',action='store_true',help='Freeze encoder layers during training')

args = parser.parse_args()

# Path to directory containing dicom files
# Expected format of these files: csv files where each line is path_to_dicom, label
normal_path = args.normal_path
svp_path = args.svp_path
fold_num = args.fold
interp_resolution = args.resolution # Resnets expect a 224x224 image
dataset_balancing_method = args.dataset_balancing
use_lr_scheduler = args.lr_scheduler

print(f'Run name: {args.run_name}')
save_model_path = args.save_model_path
print(f'Save model path: {save_model_path}')
batch_num = args.batch_size # Batch size
print(f'Fold number: {fold_num}')
print(f'Batch size: {batch_num}')
learning_rate = args.learning_rate
print(f'Learning rate: {learning_rate}')
num_epochs = args.epochs
print(f'Epochs: {num_epochs}')
print(f'Dataset balancing method: {dataset_balancing_method}')
if use_lr_scheduler:
    print('Using learning rate scheduler')
else:
    print('Not using learning rate scheduler')

pretrained = args.pretrained # Set this to True if you want to use the pretrained version
# dropout = args.dropout # Note: The foundation model always has dropout

# There are three different architectures supported: Resnet18, Resnet34, and Resnet50
architecture = args.architecture

############################################################################

#initialize dataloader. Oversample positive cases in training dataloader if specified by user
if dataset_balancing_method == 'oversampling':
    oversample = True
else:
    oversample = False

normal_pids = [] #will be list of full paths to all pid directories
for pid in os.listdir(normal_path):
    normal_pids.append(os.path.join(normal_path,pid))
svp_pids = [] #will be list of full paths to all pid directories
for pid in os.listdir(svp_path):
    svp_pids.append(os.path.join(svp_path,pid))
random.seed(10)
test_svp_pids = random.sample(svp_pids,6) #each PID should be a full path to that PID directory
test_normal_pids = random.sample(normal_pids,9)
notest_normal_pids = list(set(normal_pids) - set(test_normal_pids))
notest_svp_pids = list(set(svp_pids) - set(test_svp_pids))
random.shuffle(notest_normal_pids)
random.shuffle(notest_svp_pids)
svp_folds = []
normal_folds = []
svp_fold_size = len(notest_svp_pids) // 4
normal_fold_size = len(notest_normal_pids) // 4
for i in range(4):
    if i != 3:
        svp_folds.append(notest_svp_pids[i*svp_fold_size:(i+1)*svp_fold_size])
        normal_folds.append(notest_normal_pids[i*normal_fold_size:(i+1)*normal_fold_size])
    elif i == 3:
        svp_folds.append(notest_svp_pids[i*svp_fold_size:])
        normal_folds.append(notest_normal_pids[i*normal_fold_size:])

val_fold = svp_folds.pop(fold_num)
val_fold_labels = [1]*len(val_fold)
normal_val_fold = normal_folds.pop(fold_num)
val_fold.extend(normal_val_fold)
val_fold_labels.extend([0]*len(normal_val_fold))

train_fold = []
train_fold_labels = []
for sublist in svp_folds:
    for pid_path in sublist:
        train_fold.append(pid_path)
        train_fold_labels.append(1)
for sublist in normal_folds:
    for pid_path in sublist:
        train_fold.append(pid_path)
        train_fold_labels.append(0)

#shuffle order of training and validation folds
paired_val_folds = list(zip(val_fold,val_fold_labels))
random.shuffle(paired_val_folds)
val_fold,val_fold_labels = zip(*paired_val_folds)
val_fold = list(val_fold)
val_fold_labels = list(val_fold_labels)

paired_train_folds = list(zip(train_fold,train_fold_labels))
random.shuffle(paired_train_folds)
train_fold,train_fold_labels = zip(*paired_train_folds)
train_fold = list(train_fold)
train_fold_labels = list(train_fold_labels)

train_dataloader = DataLoader(CustomImageDataset(train_fold, train_fold_labels, True, interp_resolution, oversample), batch_size=batch_num, num_workers=2)
val_dataloader = DataLoader(CustomImageDataset(val_fold, val_fold_labels, False, interp_resolution, False), batch_size=batch_num, num_workers=2)

# Uses GPU or Mac backend if available, otherwise use CPU
# This code obtained from official pytorch docs
device = (
    f"cuda:{args.device}"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Use a Resnet model (pretrained or not depending on user input)
if architecture == 'resnet18':
    model = resnet18(pretrained=pretrained)
elif architecture == 'resnet34':
    model = resnet34(pretrained=pretrained)
elif architecture == 'resnet50':
    model = resnet50(pretrained=pretrained)
elif architecture == 'vgg11':
    model = vgg11(pretrained=pretrained)

if args.freeze_encoder:
    print('Freezing encoder layers during training')
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False
        else:
            print("Training layer: " + name)

#move model to GPU
model = model.to(device)

#Initialize loss 
if dataset_balancing_method == 'loss_weighting':
    #calculate ratio of positive to negative examples for loss weighting
    num_pos = sum(train_fold_labels)
    num_neg = len(train_fold_labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) #weight positive examples higher than negative examples since there are fewer positive examples in dataset
else:
    criterion = nn.BCEWithLogitsLoss()
#initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

with open(args.wandb_key,'r') as file:
    wandb_key = file.read().strip()

wandb.login(key=wandb_key)
run = wandb.init(project='sax-svp-classifier',
                 config={
                     "model": model,
                     "save_model_path": save_model_path,
                     "epochs": num_epochs,
                     "batch_size": batch_num,
                     "architecture": architecture 
                 },
                 name=args.run_name
                 )

# Timer
start_time = time.time()
best_auc = 0
# Training loop
for i in range(0, num_epochs):
    epoch_start_time = time.time()

    train_loop(train_dataloader, model, criterion, device, optimizer)
    test_auc = test_loop(val_dataloader, model, criterion, device)
    if test_auc > best_auc:
        best_auc = test_auc
        print(f'Saving model at epoch {i} with validation AUC of {best_auc}')
        torch.save(model.state_dict(), save_model_path)
    scheduler.step() #decrease learning rate by factor of 2 every other epoch
    elapsed_time = time.time() - epoch_start_time
    print("Epoch " + str(i + 1) + " complete at " + str(elapsed_time) + " seconds")

elapsed_time = time.time() - start_time
print('Total time: ' + str(elapsed_time) + ' seconds')