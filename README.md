Vehicle Image Classification (PyTorch)

This repo compares two different ways of solving the same problem: classifying vehicle images into four categories—Bus, Car, Truck, and Motorcycle. 

I wanted to see how much of a difference transfer learning makes on a smaller dataset, so I trained a custom CNN from scratch and compared it directly against a fine-tuned ResNet-50 using the exact same data splits and augmentations.

## The Dataset

The dataset contains 400 images split across four classes: Bus, Car, Truck, and Motorcycle. 

- Training set: 280 images[cite: 1, 2]
- Validation set: 60 images[cite: 1, 2]
- Test set: 60 images[cite: 1, 2]

To keep the models from overfitting on a relatively small training set, I applied a few standard transforms[cite: 1, 2]:
- Resized all images to 224x224[cite: 1, 2]
- Random horizontal flips and slight rotations (±15 degrees)[cite: 1, 2]
- Color jittering for brightness, contrast, and saturation[cite: 1, 2]
- Standard ImageNet normalization[cite: 1, 2]

## The Models

### 1. Custom CNN (Trained from Scratch)
For the baseline, I built a straightforward 4-layer convolutional network from the ground up[cite: 1]. Each block increases the channel depth (from 32 up to 256) with max pooling in between, followed by a 0.5 dropout layer and two fully connected layers[cite: 1]. 

- Setup: Adam optimizer (lr=0.001), Cross-Entropy loss, and early stopping (patience of 5)[cite: 1].
- Result: Early stopping kicked in at epoch 21[cite: 1]. It topped out at 63% accuracy on the test set[cite: 1]—decent for training from scratch on 280 images, though it clearly struggled to generalize on the harder edge cases[cite: 1].

### 2. ResNet-50 (Transfer Learning)
Next, I took a pre-trained ResNet-50 model, froze the early feature-extraction layers, and swapped out the final classification head with a custom dropout (0.3) and linear layer for the 4 vehicle classes[cite: 2].

- Setup: Adam optimizer (lr=0.001) paired with a ReduceLROnPlateau scheduler over 10 epochs[cite: 2].
- Result: The difference was immediate. Validation accuracy hit 98.33%, and it landed at 85% accuracy on the unseen test set[cite: 2].

## Results Comparison

| Model | Approach | Epochs | Best Val Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Custom CNN | Trained from scratch[cite: 1] | 21 (Early stopped)[cite: 1] | N/A | 63%[cite: 1] |
| ResNet-50 | Transfer learning[cite: 2] | 10[cite: 2] | 98.33%[cite: 2] | 85%[cite: 2] |

## Running the Code

1. Clone the repo and install the dependencies:
   ```bash
   pip install torch torchvision pillow numpy tqdm

## Dataset

   Get the Dataset from kaggle:
   https://www.kaggle.com/datasets/kaggleashwin/vehicle-type-recognition
