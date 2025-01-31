# This is a guide file for the entire source code file, the following is a detailed description of each file. 

## NeuroScribe Directory

  data: Normally this is a file that stores data, but due to upload restrictions and the inability to attach external links, this is an empty folder
  
  EEG_NeuroScribe_models: Outputs training logs during training and folders to save models
  
  model_test: Inside is test.ipynb, a file that holds the code for the test model, as well as the test results and visualizations. 
  
  model_train: Inside is train.py, the file used to train the model. 
  
  nets: Inside are encoder.py and model.py, which are the implementations of the TSFEEGNet model and the NeuroScribe model, respectively
  
## datapreprocess directory

It contains two files, saveasepochl. py and saveasinfo.py, which are used to process the raw EEG data, including extracting information, storing it in segments, and so on

## Auto-DMP directory

dmp_layer.py is used to implement Auto-DMP related algorithms and build models.

## load directory

The file data_loader.py is used to read the processed data during training. 

### Run saveasepoche.py and saveasinfo.py normally to process the raw EEG data andsave the processed data in the data folder. The model is then trained and saved using train.py. During the training, the log and the model are saved in EEG_NeuroScribe_models. At last, the saved model is tested with test.ipynb and the results are output
