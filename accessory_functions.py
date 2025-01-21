import pandas as pd
from pydicom import dcmread
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
from skimage.transform import rescale
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import wandb
import sys

# This function takes in a data frame of image, labels with predominantly one label
# and replicates the other labeled data to equal the length of the former
def balance_labels(labels_df):

    # Split this into the positive and negative labels
    positive_labels = []
    negative_labels = []
    for i in range(0, len(labels_df)):

        file = labels_df.iloc[i, 0]
        label = labels_df.iloc[i, 1]

        # New row to be added to negative or positive labels depending on label
        row = []
        row.append(file)
        row.append(label)

        if label == 0:
            negative_labels.append(row)
        else:
            positive_labels.append(row)

    # If either no positive or negative labels then data is not balanceable
    if len(negative_labels) == 0 or len(positive_labels) == 0:
        raise Exception("Unable to balance labels since either positive or negative labels do not exist")

    # Perform the balancing depending on which has more labels
    if len(positive_labels) <= len(negative_labels):

        # The basic idea will be to generate repeats in positive labels to match the length of
        # negative labels
        # To avoid biasing this process we first scramble the positive labels
        random.shuffle(positive_labels)

        # Repeat the elements of the positive labels
        # Note this may cause the length to exceed that of negative labels which is why we need the last line
        num_repetitions = int(np.ceil(len(negative_labels) / len(positive_labels)))
        positive_labels_extended = []
        for i in range(0, num_repetitions):
            for j in range(0, len(positive_labels)):
                row = []
                row.append(positive_labels[j][0])
                row.append(positive_labels[j][1])
                positive_labels_extended.append(row)
        positive_labels = positive_labels_extended[0:len(negative_labels)]

    else:

        # The basic idea will be to generate repeats in negative labels to match the length of
        # positive labels
        # To avoid biasing this process we first scramble the negative labels
        random.shuffle(negative_labels)

        # Repeat the elements of the negative labels
        # Note this may cause the length to exceed that of positive labels which is why we need the last line
        num_repetitions = int(np.ceil(len(positive_labels) / len(negative_labels)))
        negative_labels_extended = []
        for i in range(0, num_repetitions):
            for j in range(0, len(negative_labels)):
                row = []
                row.append(negative_labels[j][0])
                row.append(negative_labels[j][1])
                negative_labels_extended.append(row)
        negative_labels = negative_labels_extended[0:len(positive_labels)]

    # Now merge everything back into one list and convert to data frame
    labels = []
    for i in range(0, len(positive_labels)):
        labels.append(positive_labels[i])
    for i in range(0, len(negative_labels)):
        labels.append(negative_labels[i])

    # Do one final shuffle to avoid bias
    random.shuffle(labels)

    return pd.DataFrame(labels)

# Takes a path to a 2D dicom file and converts it to a tensor (with dimensions interp_resolution x interp_resolution)
def dicom_path_to_tensor(img_path,target_dim):
    # Load all dicom files into a list given the img_path
    #dicoms = [pydicom.dcmread(img_path + file) for file in os.listdir(img_path) if os.path.isfile(img_path + file)]

    # Load image dicom and conver to tensor
    dicom = dcmread(img_path)
    array = dicom.pixel_array
    #rescale to 1mmx1mm pixel size - not working currently
    # array = rescale(array, dicom.PixelSpacing,anti_aliasing=dicom.PixelSpacing[0] < 1)
    #center crop to target_dim x target_dim, with 0 padding if less
    array = array.astype(np.uint8)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad((int(np.ceil(max(target_dim-array.shape[0],0)/2)),int(np.ceil(max(target_dim-array.shape[1],0)/2)))), #can add rotation/flipping here
        transforms.CenterCrop((target_dim,target_dim))
    ])
    array = transform(array)
    array = (array - torch.mean(array)) / torch.std(array) #normalize
    return array


# Based on tutorial from PyTorch website https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    # annotations_file is a path to a csv file with the DICOM folder name, label
    # img_dir is path to folder containing all image folders, each with a collection of dicom images
    # set balance to True if you want to balance the positive and negative labels
    def __init__(self, annotations_file, target_dim=224, balance=False, transform=None, target_transform=None,file_is_df=False):

        # Note here we need to balance the positive and negative labels if train data
        if balance:
            self.img_labels = balance_labels(pd.read_csv(annotations_file, header=None))
        elif file_is_df:
            self.img_labels = annotations_file
        else:
            self.img_labels = pd.read_csv(annotations_file, header=None)

        self.transform = transform
        self.target_transform = target_transform
        self.target_dimensions = target_dim # Note we should keep this at 224x224 since that is what ResNet is built for/trained on

    def __len__(self):
        return len(self.img_labels)

    # Loads the dicoms and label of a given MRI, converts DICOMS into a 3Dntensor,
    # interpolates to a given size, and returns image tensor and label
    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path = self.img_labels.iloc[idx, 0]
        image = dicom_path_to_tensor(img_path,self.target_dimensions)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.tensor([label],dtype=torch.float32)


def train_loop(dataloader, model, loss_fn, device, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    current_loss = 0
    all_pred_proba = []
    all_y = []
    for batch in range(len(dataloader)):

        (X, y) = next(iter(dataloader))

        # Move X and y to GPU
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        current_loss += loss.item()
        sigmoid = nn.Sigmoid()
        pred_sigmoid = sigmoid(pred) 
        pred_np = pred_sigmoid.cpu().detach().numpy().squeeze()
        y_np = y.cpu().detach().numpy().squeeze()
        all_pred_proba.extend(pred_np)
        all_y.extend(y_np)
        if (batch+1) % 100 == 0 or (batch+1) == len(dataloader):
            try:
                auc = roc_auc_score(all_y,all_pred_proba)
                all_pred = np.round(all_pred_proba)
                recall = recall_score(all_y,all_pred)
                precision = precision_score(all_y,all_pred)
                accuracy = accuracy_score(all_y,all_pred)
                specificity = recall_score(all_y,all_pred,pos_label=0)
                train_loss = current_loss/(batch + 1)
                print(f"Batch: [{batch+1:>5d}/{len(dataloader):>5d}]")
                print(f"Train Error: \n   Accuracy: {accuracy:>0.3f}\n   recall: {recall:>0.3f}\n   specificity: {specificity:>0.3f}\n   precision: {precision:>0.3f}\n   AUC: {auc:>0.3f}\n   Avg loss: {train_loss:>8f} \n")
                wandb.log({'train_loss':train_loss})
                wandb.log({'train_loss': train_loss,"train_acc":accuracy,"train_precision":precision,"train_recall":recall,"train_specificity":specificity,"train_auc":auc})
            except:
                print('ERROR! see error_y.txt and error_pred_proba.txt')
                with open('error_y.txt','w') as file:
                    for val in all_y:
                        file.write(f'{val}\n')
                with open('error_pred_proba.txt','w') as file:
                    for val in all_pred_proba:
                        file.write(f'{val}\n')
                sys.exit()

def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    all_pred_proba = []
    all_y = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for (X, y) in dataloader:

            # First move X and y to GPU
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            sigmoid = nn.Sigmoid()
            pred_sigmoid = sigmoid(pred) 
            pred_np = pred_sigmoid.cpu().numpy().squeeze()
            y_np = y.cpu().numpy().squeeze()
            all_pred_proba.extend(pred_np)
            all_y.extend(y_np)

    test_loss /= num_batches
    auc = roc_auc_score(all_y,all_pred_proba)
    all_pred = np.round(all_pred_proba)
    recall = recall_score(all_y,all_pred)
    precision = precision_score(all_y,all_pred)
    accuracy = accuracy_score(all_y,all_pred)
    specificity = recall_score(all_y,all_pred,pos_label=0)
    print(f"Test Error: \n   Accuracy: {accuracy:>0.3f}\n   recall: {recall:>0.3f}\n   specificity: {specificity:>0.3f}\n   precision: {precision:>0.3f}\n   AUC: {auc:>0.3f}\n   Avg loss: {test_loss:>8f} \n")
    wandb.log({'test_loss': test_loss,"test_acc":accuracy,"test_precision":precision,"test_recall":recall,"test_specificity":specificity,"test_auc":auc})
    return all_pred, all_y
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       