import numpy as np
from tqdm import tqdm

class LogisticRegression:
    """
    Binary Logistic Regression using Gradient Descent
    """
    def __init__(self, epoch=500, lr=0.01):
        self.epoch = epoch
        self.lr = lr
        self.weights = None
        self.losses = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def _loss_fn(self, y, y_hat):
        """Binary cross-entropy loss"""
        epsilon = 1e-8
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss
    
    def fit(self, X, y):
        """
        Train the model using Gradient Descent
        
        Parameters:
        X: numpy array of shape (n_samples, n_features) - training data
        y: numpy array of shape (n_samples, 1) - training labels (0 or 1)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        
        for epoch in tqdm(range(self.epoch), desc="Training"):
            # Forward pass
            z = np.dot(X, self.weights)
            y_pred = self._sigmoid(z)
            
            # Compute loss
            loss = self._loss_fn(y, y_pred)
            self.losses.append(loss)
            
            # Backward pass
            gradient = (1/n_samples) * np.dot(X.T, (y_pred - y))
            self.weights -= self.lr * gradient
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = np.dot(X, self.weights)
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict binary labels"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)