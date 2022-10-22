# Concrete Crack Detection

## Problem Statement

Buildings require regular maintenance and retrofitting works in order to remain structurally safe and to be in line with new building regulations. 

Crack are one of the major problems with concrete structures, mainly caused by poor construction practices, corrosion of reinforcements in concretes and improper structural designs and specifications etc. Crack detection plays a major role in building inspection. 

As most building inspection are done manually, the quality of inspection might deteriorate over time and human might fail to identify detrimental cracks in buildings.

This project aims to use convolutional neural network models (CNN) to classify concrete images with or without cracks.

## About the Dataset 

The datasets contains images of various concrete surfaces with and without crack. The image data are divided into two as negative (without crack) and positive (with crack) in separate folder for image classification. Each class has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels. The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016). High resolution images found out to have high variance in terms of surface finish and illumination condition. No data augmentation in terms of random rotation or flipping or tilting is applied.

The dataset can be found from the website [Mendeley Data - Crack Detection](https://data.mendeley.com/datasets/5y9wdsg2zt/2), contributed by Çağlar Fırat Özgenel.

## Modelling 

A total of 3 models are being modelled, convolutional neural networks (CNN), transfer learning with VGG16 and transfer learning with InceptionV3. The CNN model is a simple CNN typical architecture, having 3 layers of Conv2D and MaxPool2D, followed by GlobalAveragePooling2D, 30% dropout and output dense layer with sigmoid activation. For the transfer learning VGG16 model, the dense layers are removed and replace with 1 layer of GlobalAveragePooling2D, 1 dense layer with 512 neurons, 30% dropout and output dense layer is the same as the CNN output layer. Lastly, for the transfer learning InceptionV3 model, the last few fully connected layer are removed and replace with the same layer structure applied for the VGG16 model with 1024 neurons instead of 512. 


## Results/Evaluation of Models

Model | loss | val_loss | test_loss | train_accuracy | val_accuracy | test_accuracy|
--------|-----|-----|-----|-----|-----|-----|
CNN                           | 0.0214 | 0.0163 | 0.0120 | 99.36% | 99.58% | 99.67% |
Transfer-Learning VGG16       | 0.0053 | 0.0055 | 0.0047 | 99.84% | 99.86% | 99.90% |
Transfer-Learning InceptionV3 | 0.0038 | 0.0215 | 0.0023 | 99.88% | 99.47% | 99.92% |

Generally, all three models are performing well with an accuracy of above 95%, with a slight underfitting of models (consider reducing the training set e.g 80-20 split to 70-30). Transfer learning model using VGG16 seems to perform the best with the smallest generalisation between training loss and validation loss, train and validation accuracy. Therefore, even though InceptionV3 has the highest test accuracy, VGG16 model is preferred with a slight compromisation in test accuracy of 0.02% in comparision with InceptionV3 test accuracy. In additional, VGG16 model predicts less false negative images as compared inceptionV3 model. 

## Conclusion

In conclusion, all three models perform fairly well with accuracy of ~99%. Transfer learning model using VGG16 performs the best as it has the smallest generalisation between training loss and validation loss amount all three models. However it seems that the model could only predict cracks in concrete and close-up images of the cracks. The model could not predict -- 
- high-resolution images for images of crack from afar (but these images can be divided into smaller pieces and identify whether crack exist)
- Other non-concrete cracks such as cracks in facades or asphalt road


### Recommendations

1. Predicting whether the the cracks are structural or non-structural cracks could be an interesting topic to research on as this will help people to identify whether the cracks will affect the structural integrity of the building or determine whether the building is under-designed. Datasets should include images of structural cracks and usually these are not close up images. Location of the crack is needed to determine whether there are structural cracks and type of structural cracks -- be it cracks formed on the beam, column, wall or slab. 

2. Modern buildings comes with different kind of facades, most commonly glass, steel facades. We might also want to research on the cracks in these facades of different material from concrete. However, do take note that green facades might affect model's prediction and look into how to overcome the difficulty in identify cracks for building with green facades.
