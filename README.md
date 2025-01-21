# Short Axis SVP Classifier 
This repository contains scripts to train a binary classifier to identify hearts as single ventricle physiology (SVP) or normal based on short axis cardiac MRI images. The basic usage is described below. Necessary python libraries that will need to be installed (excluding pytorch) are contained in requirements.txt. Pytorch should be installed with CUDA support following the instructions at https://pytorch.org/

## Basic Usage
main.py:  
Primary script to start training runs. Necessary arguments include:  
- --labels_train: Path to a csv containing a list of all images in the training dataset and the image label (1 for SVP, 0 for normal)  
- --labels_test: Path to a csv containing a list of all images in the testing dataset and the image label (1 for SVP, 0 for normal)  
- --run_name: Name to identify run in Weights and Biases (WandB)  
- --wandb_key: Path to file containing WandB login key  
- --save_model_path: Path to save final model to  
- --architecture: (Choices are ['resnet18','resnet34','resnet50']) Selects which ResNet architecture to use in training.  
- --device: Cuda GPU ID  
Optional arguments:  
- --pretrained: Initialize ResNet model with default PyTorch pretrained weights  
- --epochs: Number of epochs (default=3)  
- --resolution: Resolution to resize images to (default=224)  
- --batch_size: Set batch size (default=50)  
- --learning_rate: Sets learning rate (default=0.0001)  
For additional help, run `python main.py -h` for a list of all parameters  
Example execution  
`python main.py --labels_train data_paths/fold_0_train.csv --labels_test data_paths/fold_0_test.csv --architecture resnet18 --save_model_path saved_models/res18_fold0_weightedloss.pth --run_name res18_fold0_weightedloss --wandb_key wandb_key.txt --device 0`

### Other Scripts
models.py:  
    Contains functions to create ResNet models modified to accept only 1 color channel as input and to output only a single class prediction

accessory_functions.py:  
    Contains functions to preprocess images, generate dataset objects, oversample positive examples, and run train and test loops

gen_analysis_csv.py:  
    Script to use trained model weights and a test dataset to generate a CSV file with model scores on each test image organized by patient ID, z-axis coordinate, and time stamp.
    For usage instructions, run `python gen_analysis_csv.py -h`

## Further details on dataset
This dataset contains data from 267 patients who received short axis cardiac MRI imaging at UCLA, totalling 673 unique scans. 54 patients have SVP (148 unique scans), and 213 patients do not (525 unique scans). To increase the size of training data, the dataset is supplemented by data from ~1000 patients from the Kaggle Second Annual Data Science Bowl (https://www.kaggle.com/c/second-annual-data-science-bowl).  
Training is done on each 2D slice from each scan. In the future, a 3D model will be developed. 

## Training procedure
- Pre-processing: Images are padded then center-cropped to a size of target_dimension x target_dimension (224x224 by default). Pixel values are rescaled to be between [0,1], and then normalized by subtracting the mean pixel value and dividing by the standard deviation.  
- Cross-validation splits are done beforehand by separating the UCLA and Kaggle data into groups of all positive examples and all negative examples (all Kaggle data are negative), then randomly separating into 4 equal size groups based on patient ID to create 4 folds for cross-validation.  
- main.py is used as described above to begin training  
  - Models are created based on the ResNet architecture (https://arxiv.org/abs/1512.03385), modified to accept only 1 color channel in inputs and to predict the probability for only one class for binary classification (i.e. SVP or not).  
- Models are evaluated by writing evaluation metrics (loss, AUROC, accuracy, precision, specificity, sensitivity) of training and validation sets to Weights and Biases during training.  
After all epochs are completed, model weights are saved and further analysis based on patient ID, z-axis of the imaging slices, or time stamps using the gen_analysis_csv.py script.  

## Results
![image](https://github.com/user-attachments/assets/d412c107-73a3-45e1-bf04-2b4255953844)

![image](https://github.com/user-attachments/assets/1e487f70-ccbb-456d-b780-9239a0ab2486)

![image](https://github.com/user-attachments/assets/c0aeff51-d6c7-48f5-b2e4-0ab7a443af9e)

![image](https://github.com/user-attachments/assets/9eecb5b0-d8c5-4ff4-ad12-a18ea1949526)


## Next steps
Performance of the model on 2D slices alone has been unimpressive, and it is unlikely that 2D slices alone contain enough information to diagnose SVP. Next steps include:  
- creating a 3D model to incorporate more of the heart's structures in the prediction
  - restructure data file storage organization
- improving data augmentation (do all permutations of augmentation instead of random)
- including relevant metrics derived from automated segmentations such as ventricular volumes, ejection fraction, and myocardial mass
