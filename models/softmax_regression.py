import numpy as np
from tqdm import tqdm

class SoftmaxRegression:
    """
    Softmax Regression (Multiclass) using Gradient Descent
    """
    def __init__(self, epoch=500, lr=0.01, num_classes=10):
        self.epoch = epoch
        self.lr = lr
        self.num_classes = num_classes
        self.weights = None
        self.losses = []
    
    def _softmax(self, z):
        """Softmax activation function"""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _loss_fn(self, y_onehot, y_pred):
        """Cross-entropy loss"""
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_onehot * np.log(y_pred), axis=1))
        return loss
    
    def fit(self, X, y_onehot):
        """
        Train the model using Gradient Descent
        
        Parameters:
        X: numpy array of shape (n_samples, n_features) - training data
        y_onehot: numpy array of shape (n_samples, num_classes) - one-hot labels
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, self.num_classes))
        
        for epoch in tqdm(range(self.epoch), desc="Training"):
            # Forward pass
            z = np.dot(X, self.weights)
            y_pred = self._softmax(z)
            
            # Compute loss
            loss = self._loss_fn(y_onehot, y_pred)
            self.losses.append(loss)
            
            # Backward pass
            gradient = (1/n_samples) * np.dot(X.T, (y_pred - y_onehot))
            self.weights -= self.lr * gradient
    
    def predict_proba(self, X):
        """Predict probabilities for each class"""
        z = np.dot(X, self.weights)
        return self._softmax(z)
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)