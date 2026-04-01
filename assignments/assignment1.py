import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from models.logistic_regression import LogisticRegression
from utils.data_loader import load_mnist_data
from utils.data_preprocessor import preprocess_binary
from utils.metrics import calculate_binary_metrics

def main():
    print("="*60)
    print("ASSIGNMENT 1: Binary Logistic Regression (Digits 0 vs 1)")
    print("="*60)
    
    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Preprocess
    X_train, y_train = preprocess_binary(train_images, train_labels)
    X_test, y_test = preprocess_binary(test_images, test_labels)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train model
    model = LogisticRegression(epoch=500, lr=0.01)
    model.fit(X_train, y_train)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(model.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss - Binary Logistic Regression')
    plt.grid(True, alpha=0.3)
    #plt.savefig('results/loss_curves/logistic_loss.png')
    plt.show()
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = calculate_binary_metrics(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()