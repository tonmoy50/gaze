# MultiNet Gaze Predictor

## Setup
The project is dependent on python environment and all the packages associated is built based-on python version 3.11.

In order to setup the dependencies run the following command in your python environment command line,
```pip install -r requirements.txt```

This should install all the required packages.

## Dataset
You can download the dataset from the following link - https://drive.google.com/drive/folders/1AKA1jCVdrMmLIXqTeNNCFo5VnUrAcplq

## Training
Make sure you have the dataset in your system and is accessible. Update the ```FILE_ROOT``` in ```dataset.py``` with you dataset location root directory first. \
Then in order to train the model run the following file from the root of the project - ```python main.py```
\
The training instance will output the losses in a ```txt``` file in same directory.