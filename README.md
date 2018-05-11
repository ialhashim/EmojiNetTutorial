# EmojiNetTutorial
A tutorial on AI to solve the problem of object detection and segmentation using Keras. We use a modified U-Net to perform a basic instance segmentation on 128px targets.

## 1) Prepare datasets
To generate the dataset for training or testing you first need to run:
```
python prepare_dataset.py
```

## 2) Training
Once the datasets are ready, train the model by running:
```
python train.py
```

## 3) Testing
You can generate novel random scences by using the same functions in the ```prepare_dataset.py```. Then you can either a command line or GUI testing application by running:
```
python test_gui.py
```
or:
```
python test.py
```

## Misc.
The file ```environment.yml``` is a conda environment file. The ```dataset.py``` script loads and resample the data to the desired resolutoin.
