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
import os
import sys
import statistics

# Takes a path to a 2D dicom file and converts it to a tensor (with dimensions interp_resolution x interp_resolution)
def dicom_path_to_tensor3d(img_path,target_dim,train=True):
    # Load image dicoms contained in img_path, put into 3d volume, pre-process into tensor for input into model
    img = []
    if train:
        transform_augmentation = transforms.Compose([
            transforms.RandomRotation(degrees=(0,180)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
    for slice_path in img_path:
        dicom = dcmread(slice_path)
        array = dicom.pixel_array
        #rescale to 1mmx1mm pixel size - not working currently
        # array = rescale(array, dicom.PixelSpacing,anti_aliasing=dicom.PixelSpacing[0] < 1)
        #center crop to target_dim x target_dim, with 0 padding if less
        array = array.astype(np.uint8)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1181772], std=[0.13326852023601532]),
            transforms.Pad((int(np.ceil(max(target_dim-array.shape[0],0)/2)),int(np.ceil(max(target_dim-array.shape[1],0)/2)))), #can add rotation/flipping here
            transforms.CenterCrop((target_dim,target_dim)),
        ])
        array = transform(array)
        #apply image augmentation if training
        if train:
            array = transform_augmentation(array)
        img.append(array)
    img = torch.stack(img) # (5x1x224x224)
    img = img.squeeze() # (5x224x224)
    return img


# Based on tutorial from PyTorch website https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    # directory structure is: 
    # label
    # ---pid
    # ------date/title
    # ---------sequences (are each of this unique runs, or are they organized by z-axis/time stamp? )
    # all the examples in the given folder for training should have the same label
    # paths should contain the path to all the relevant PID folders
    # set balance to True if you want to use oversampling balance the positive and negative labels
    def __init__(self, pids, labels, train=True, target_dim=224, balance=False):
        # if balance:
        #     self.img_labels = balance_labels(pd.read_csv(annotations_file, header=None)) # this won't work anymore
        self.all_paths = []
        self.all_labels = []
        self.num_skipped = 0
        for i,pid_path in enumerate(pids):
            pt_images = self.get_images(pid_path)
            self.all_paths.extend(pt_images)
            self.all_labels.extend([labels[i]]*len(pt_images))
        print(f"Number skipped: {self.num_skipped}")
        if balance:
            self.balance_dataset()
        #shuffle order of paths and labels together
        paired_paths_labels = list(zip(self.all_paths,self.all_labels))
        random.shuffle(paired_paths_labels)
        all_paths,all_labels = zip(*paired_paths_labels)
        self.all_paths = list(all_paths)
        self.all_labels = list(all_labels)
        self.train = train #store whether this is a training set or a validation/test set
        self.target_dimensions = target_dim # Note we should keep this at 224x224 since that is what ResNet is built for/trained on

    def balance_dataset(self):
        positive_paths = [path for path, label in zip(self.all_paths, self.all_labels) if label == 1]
        negative_paths = [path for path, label in zip(self.all_paths, self.all_labels) if label == 0]
        
        num_positive = len(positive_paths)
        num_negative = len(negative_paths)
        
        if num_positive < num_negative:
            positive_paths = positive_paths * (num_negative // num_positive) + positive_paths[:num_negative % num_positive]
        elif num_negative < num_positive:
            negative_paths = negative_paths * (num_positive // num_negative) + negative_paths[:num_positive % num_negative]
        
        self.all_paths = positive_paths + negative_paths
        self.all_labels = [1] * len(positive_paths) + [0] * len(negative_paths)
    
    def get_images(self,pid_path):
        #write function that based on patient ID path returns all 3d image stacks
        image_paths = []
        date_paths = os.listdir(pid_path)
        for date_path in date_paths:
            pid_date_path = os.path.join(pid_path,date_path)
            num_locs = len(os.listdir(pid_date_path))
            if num_locs < 5: #skip image if there is not enough z-axis locations (5 here)
                self.num_skipped += 1
                continue
            all_sequences = os.listdir(pid_date_path)
            all_locs = []
            num_times = [] #used to remove sequences where the number of time stamps differs from the other sequences in a series
            #acquire all locs
            for sequence in all_sequences:
                seq_path = os.path.join(pid_date_path,sequence)
                num_times.append(len(os.listdir(seq_path))) 
                for dcm in os.listdir(seq_path):
                    dcm_path = os.path.join(seq_path,dcm)
                    dcm = dcmread(dcm_path)
                    loc = float(dcm.get((0x0020,0x1041),'Unknown').value)
                    all_locs.append(loc)
                    break
            has_unique = False
            for i in range(len(num_times)):
                for j in range(len(num_times)):
                    if num_times[i] != num_times[j]:
                        has_unique = True
                        break
            if has_unique:
                #remove all sequence and loc references to the sequence directories of the minority class
                mode_time = statistics.mode(num_times)
                remove_indexes = [i for i,num in enumerate(num_times) if num != mode_time]
                all_sequences = [seq for i,seq in enumerate(all_sequences) if i not in remove_indexes]
                all_locs = [loc for i,loc in enumerate(all_locs) if i not in remove_indexes]
                if len(all_locs) < 5: #check again that we have enough z-axis coordinates after filtering
                    self.num_skipped += 1
                    continue
            #sort sequences according to loc
            combined_list = list(zip(all_locs,all_sequences))
            sorted_combined_list = sorted(combined_list)
            all_locs, all_sequences = zip(*sorted_combined_list)
            all_locs = list(all_locs)
            all_sequences = list(all_sequences)
            #grab 5 slices distributed approximately evenly throughout volume
            keep_indices = [0,len(all_sequences)//3,len(all_sequences)//2,len(all_sequences)*2//3,-1]
            keep_sequences = []
            for i in keep_indices:
                keep_sequences.append(all_sequences[i])
            #create image stacks of 5 locs at all time points
            final_paths = []
            for sequence in keep_sequences:
                seq_path = os.path.join(pid_date_path,sequence)
                all_times = []
                all_paths = []
                for dcm in os.listdir(seq_path):
                    dcm_path = os.path.join(seq_path,dcm)
                    dcm = dcmread(dcm_path)
                    time = float(dcm.TriggerTime)
                    all_times.append(time)
                    all_paths.append(dcm_path)
                combined_list = list(zip(all_times,all_paths))
                sorted_combined_list = sorted(combined_list)
                all_times, all_paths = zip(*sorted_combined_list)
                all_times = list(all_times)
                all_paths = list(all_paths) # list of paths for dicoms at all time points at certain z-axis
                final_paths.append(all_paths) # list of 5 lists of dicoms at all time points at certain z-axis
            final_paths = np.array(final_paths).T.tolist() #list with all time points where each element contains 5 paths to dicom files sorted by z-axis from small to large
            image_paths.extend(final_paths)
        #list shape check, can delete if functions properly TESTING
        for img in image_paths:
            if len(img) != 5:
                print(f"Improper image length of: {len(img)}")
                print(img)
        return image_paths

    def __len__(self):
        return len(self.all_paths)

    # Loads the dicoms and label of a given MRI, converts DICOMS into a 3Dntensor,
    # interpolates to a given size, and returns image tensor and label
    def __getitem__(self, idx):
        img_path = self.all_paths[idx]
        image = dicom_path_to_tensor3d(img_path,self.target_dimensions,self.train)
        return image, torch.tensor([self.all_labels[idx]],dtype=torch.float32)


def train_loop(dataloader, model, loss_fn, device, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    current_loss = 0
    all_pred_proba = []
    all_y = []
    for batch, (X,y) in enumerate(dataloader):
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
    return auc
       