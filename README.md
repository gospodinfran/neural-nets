### Neural network models increasing in complexity & capabilities

# Bigram model 

One character of context predicts the next.

# Image classification model

-- FashionMNIST 28x28 greyscale images.

-- MNIST digits dataset.

Some of the neural nets:

## Feedforward neural net

Feedforward neural net which achieved 90% accuracy on a binary image fashionMNIST dataset. 

![FeedforwardNN](/assets/Feedworward%20NN.png)

## Convolutional neural net 

Convolutional neural net which consists of two parts: convolutional net and a fully connected net. In simple terms: the convolutional net extracts features, while the fully connected network makes predictions based on the extracted features.

![CNN](/assets/CNN_model.png)

### Locally-saved weights

Weights can also be saved locally, instead of only in memory. Note: you can only load weights if state dict (weights) is  initialized and/or trained. 

![LoadWeights](/assets/LoadNSaveWeights.png)