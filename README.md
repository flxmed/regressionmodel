## California Housing Price Prediction
A neural network regression model built from scratch in PyTorch to predict median house values.


## Overview
I built a model from scratch using PyTorch, pandas for data processing, and NumPy. I came across problems like overfitting(memorization of training data instead of learning it), missing data(total_bedrooms had null values), feature scale mismatch(features had wildly different scales) and learning rate decay(fixed learning rates can get stuck)


## Architecture & decisions
To fix overfitting problem I've decided to use regularization technique called dropout with value of (p=0.2). Dropout basically deactivates a fraction of neurons(0.2=20%) preventing neurons from co-adapting and forcing the network to learn more robust features.  My model had 3 layers with hidden_size=128 to provide flexibility for training data providing enough capacity to model non-linear relationships in the data. As optimizer I used Adam. Adam is an adaptive optimizer that extends gradient descent with momentum and per-parameter learning rates, leading to faster and more stable convergence. Also ocean_proximity is a categorical feature with 5 unique values. Since neural networks require numerical input, I implemented manual one hot encoding - converting each category into a binary column. Target variable median_house_value was normalized to stabilize gradient updates during training


## Results
Final Test RMSE: $48,420


## How to run
pip install -r requirements.txt

python housing_regression.py
