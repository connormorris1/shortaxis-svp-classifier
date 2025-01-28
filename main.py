import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import time
import wandb
import argparse
from accessory_functions import CustomImageDataset, train_loop, test_loop
from models import resnet18, resnet34, resnet50, vgg11

############################################################################
parser = argparse.ArgumentParser(description='Begin training runs')
parser.add_argument('--labels_train',type=str,required=True,help='path to training labels')
parser.add_argument('--labels_test', type=str,required=True,help='path to testing labels')
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

args = parser.parse_args()

# Path to directory containing dicom files
# Expected format of these files: csv files where each line is path_to_dicom, label
labels_train = args.labels_train
labels_test = args.labels_test
interp_resolution = args.resolution # Resnets expect a 224x224 image
dataset_balancing_method = args.dataset_balancing

print(f'Run name: {args.run_name}')
save_model_path = args.save_model_path
print(f'Save model path: {save_model_path}')
batch_num = args.batch_size # Batch size
print(f'Batch size: {batch_num}')
learning_rate = args.learning_rate
print(f'Learning rate: {learning_rate}')
num_epochs = args.epochs
print(f'Epochs: {num_epochs}')
print(f'Dataset balancing method: {args.dataset_balancing}')

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
train_dataloader = DataLoader(CustomImageDataset(labels_train, interp_resolution, oversample), batch_size=batch_num)
test_dataloader = DataLoader(CustomImageDataset(labels_test, interp_resolution, False), batch_size=batch_num)

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
    model = resnet50(pretrained=pretrained)

#move model to GPU
model = model.to(device)

#Initialize loss 
if dataset_balancing_method == 'loss_weighting':
    #calculate ratio of positive to negative examples for loss weighting
    training_df = pd.read_csv(labels_train,header=None)
    num_pos = len(training_df[training_df[1] == 1])
    num_neg = len(training_df[training_df[1] == 0])
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) #weight positive examples higher than negative examples since there are fewer positive examples in dataset
else:
    criterion = nn.BCEWithLogitsLoss()
#initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

# Training loop
for i in range(0, num_epochs):
    epoch_start_time = time.time()

    train_loop(train_dataloader, model, criterion, device, optimizer)
    test_loop(test_dataloader, model, criterion, device)

    elapsed_time = time.time() - epoch_start_time
    print("Epoch " + str(i + 1) + " complete at " + str(elapsed_time) + " seconds")

elapsed_time = time.time() - start_time
print('Total time: ' + str(elapsed_time) + ' seconds')

# Save our model
# See this tutorial for how to load our model: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
torch.save(model.state_dict(), save_model_path)