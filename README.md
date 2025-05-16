# Cats vs. Dogs: Autoencoder-Based Classifier
 
## Overview   
#### In this project I compared three different approaches for binary image classification on a labeled dataset of cat and dog images obtained from a previous Kaggle competition: https://www.kaggle.com/c/dogs-vs-cats. The training folder contained 25,000 images of dogs and cats. I split the labeled images into training and validation sets with an 80/20 ratio to evaluate model performance since the test data were unlabeled. Predictions were made for the labels of images in the test dataset (test1.zip), i.e., 1 = dog, 0 = cat.

## Motivation
#### The goal was to explore unsupervised feature learning via autoencoders and evaluate how well such learned features support image classification compared to a purely supervised CNN. This helped deepen my understanding of representation learning and model interpretability.

## Installation
#### Clone the repository and install required packages:
```bash
git clone https://github.com/gtg975n/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
pip install -r requirements.txt

 ```   
 
## Usage
#### Launch the main Jupyter notebook cats_vs_dogs_autoencoder.ipynb to reproduce training, evaluation, and visualization.

#### Alternatively, explore the project/ folder for modular scripts and models.

## Results Summary
#### ResNet18 achieved the highest classification accuracy, serving as a strong supervised baseline.

#### The Autoencoder + classifier pipeline captured important image features like edges and contours but lagged in accuracy.

#### Joint training of the autoencoder and classifier improved classification performance compared to the two-stage approach.

#### Difference plots highlighted that the autoencoder primarily learned edge and contour information, which are key for downstream classification.

### Citation   
#### If you use this project, please cite:
```
@misc{mymccatsdogs2025,
  author = {George McConnell},
  title = {Cats vs. Dogs: Autoencoder-Based Image Classification},
  year = {2025},
  note = {GitHub repository: https://github.com/gtg975n/cats-vs-dogs-classifier}
}

```   
