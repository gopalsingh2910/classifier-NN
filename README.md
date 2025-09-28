# Neural Network Classifier

## Overview

This project implements a neural network-based image classifier designed to distinguish between three distinct animal categories: **Cats**, **Dogs**, and **Snakes**. The model is built using a fully connected (Dense) architecture and trained on image data.

## Configuration and Architecture

The model's behavior and structure are defined in the `config.py` file.

### Key Configuration Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `input_dim` | 128 | Images are resized to $128 \times 128$ pixels. |
| `num_classes` | 3 | Corresponds to the three classes: Cat, Dog, Snake. |
| `batch_size` | 100 | The number of samples processed per iteration. |
| `num_epochs` | 100 | The total number of training cycles. |
| `learning_rate` | 0.005 | The step size for gradient descent optimization. |
| `hidden_layer1_size` | 64 | Neurons in the first hidden layer. |
| `hidden_layer2_size` | 32 | Neurons in the second hidden layer. |
| `dropout_rate` | **0.5** | Probability of setting an activation to zero for regularization. |

### Neural Network Structure

The architecture is a simple Deep Neural Network (DNN) with two hidden layers:

1.  **Input Layer:** $128 \times 128 \times 3 = 49152$ features (flattened image data).
2.  **Hidden Layer 1:** 64 Neurons (e.g., ReLU activation).
3.  **Dropout Layer:** Rate **0.5** (applied after the first hidden layer).
4.  **Hidden Layer 2:** 32 Neurons (e.g., ReLU activation).
5.  **Output Layer:** 3 Neurons (Softmax activation).

---

## Performance Results: Identifying Overfitting

The initial training results show a clear case of **overfitting**, where the model has learned the training data exceptionally well but performs poorly on unseen data.

| Metric | Result |
| :--- | :--- |
| **Training Accuracy** | **97.21%** |
| **Test Accuracy** | **53.33%** |

### Analysis

The dropout of 0.5 prevents model from overfitting but from stats it seems like it is not sufficient.
The significant gap between the training accuracy ($\approx 97\%$) and the test accuracy ($\approx 53\%$) indicates that the model has **memorized** the training set's noise and specific examples.

## Training Visualization

The following plot illustrates the loss history and convergence during the training process. A typical sign of overfitting here would be the training loss continuing to decrease while the validation/test loss begins to increase.

![Loss History Statistics](https://github.com/gopalsingh2910/classifier-NN/blob/main/loss_history_stats.png)

---

## Mitigation Strategies

To achieve a higher and more balanced test accuracy, the following steps are prioritized:

1.  **Switch to CNN Architecture:** For image data, **Convolutional Neural Networks (CNNs)** are standard and highly recommended, as they excel at extracting spatial features and are generally more robust to image variations.
2.  **Aggressive Regularization:**
    * **Increase Dropout:** Try increasing the dropout rate slightly (e.g., to $0.6$ or $0.7$) or adding dropout to the second hidden layer.
    * **L1/L2 Regularization:** Add L1 or L2 weight decay to the hidden layers.
3.  **Data Augmentation:** Implement techniques like random rotation, flipping, etc.

## Conclusion

The current model successfully achieved near-perfect performance on the training set, but its low test accuracy highlights a critical **generalization problem** due to overfitting. Future development will focus on implementing a **CNN architecture** and employing stronger regularization techniques to close the gap between training and testing performance, aiming for a robust and reliable classifier for cats, dogs, and snakes.
