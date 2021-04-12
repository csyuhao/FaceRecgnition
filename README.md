# Face Recognition

## Description

This repo is the face recognition system, which consists of the face detector, the face extractor and the classifier. The face detector is utilized to localize the face in a image, the face extractor is applied to encode the detected face image into the 512-dim feature vectors, and the classifier is employed to classify the feature vector.

## File Structure

This repo consists of three different modules, which is described as below.

```
|- backbone         The network structure of feature extractor
|- dataset          The class for loading face images
|- loss             The loss function for training the feature extractor
|- preprocess       The feature extractor
|- util             The util functions for visualize
|- video            The demo of the face recognition systems        
```

**How to train the feature extractor?**

```Bash
conda activate env-name
python train.py --train_root=datasets\CASIA-WebFace-112x112 --train_file_list=datasets\CASIA-WebFace-112x112.list --lfw_test_root= datasets\lfw-112x112 --lfw_file_list=datasets\pairs.txt
```

**How to fine-tuning the face recognition system?**
```Bash
conda activate env-name
python finetuning.py --train_root=Dataset\Human_Face_Dataset\facebank-112x112 --train_file_list=Dataset\Human_Face_Dataset\facebank-112x112.list
```