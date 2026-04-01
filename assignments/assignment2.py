import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from models.softmax_regression import SoftmaxRegression
from utils.data_loader import load_mnist_data
from utils.data_preprocessor import preprocess_multiclass
from utils.metrics import calculate_multiclass_metrics

def main():
    print("="*60)
    print("ASSIGNMENT 2: Softmax Regression (Digits 0-9)")
    print("="*60)
    
    # 1. Load data
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # 2. Preprocess (Flatten, Normalize, Add Bias, One-hot encoding)
    X_train, y_train_onehot = preprocess_multiclass(train_images, train_labels, num_classes=10)
    X_test, y_test_onehot = preprocess_multiclass(test_images, test_labels, num_classes=10)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # 3. Train model
    model = SoftmaxRegression(epoch=500, lr=0.01, num_classes=10)
    model.fit(X_train, y_train_onehot)
    
    # 4. Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(model.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss - Softmax Regression (Digits 0-9)')
    plt.grid(True, alpha=0.3)
    # plt.savefig('results/loss_curves/softmax_loss.png')
    plt.show()
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    metrics = calculate_multiclass_metrics(test_labels, y_pred)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:                  {metrics.get('accuracy', np.mean(y_pred == test_labels)):.4f}")
    
    # Hiển thị thêm các metrics nâng cao nếu dictionary metrics hỗ trợ
    if 'macro' in metrics:
        print(f"  Macro Average - Precision: {metrics['macro']['precision']:.4f}")
        print(f"  Macro Average - Recall:    {metrics['macro']['recall']:.4f}")
        print(f"  Macro Average - F1:        {metrics['macro']['f1']:.4f}")
        print(f"  Micro Average - Precision: {metrics['micro']['precision']:.4f}")
        print(f"  Micro Average - Recall:    {metrics['micro']['recall']:.4f}")
        print(f"  Micro Average - F1:        {metrics['micro']['f1']:.4f}")

    # 6. Sample Predictions
    print("\n--- Sample Predictions (First 10 images) ---")
    for i in range(min(10, len(X_test))):
        true_label = test_labels[i]
        pred_label = y_pred[i]
        status = "✓" if true_label == pred_label else "✗"
        print(f"  Sample {i+1:2d}: True={true_label}, Pred={pred_label} {status}")

    print("\n" + "="*60)
    print("✅ ASSIGNMENT 2 HOÀN THÀNH!")
    print("="*60)

if __name__ == "__main__":
    main()