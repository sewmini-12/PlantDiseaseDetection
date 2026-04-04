import numpy as np
import cv2
import os
import time
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import process_image

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
print(f"Loaded SVM training data: {X_train.shape}, {y_train.shape}")

print("\n--- Training SVM (Traditional CV Path) ---")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
joblib.dump(svm, 'plant_disease_svm.joblib')
print("SVM training complete and saved as 'plant_disease_svm.joblib'.")


test_dir = "split_data/test"
classes = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]
class_to_label = {cls: i for i, cls in enumerate(classes)}

X_test = []
y_test = []
inference_times = []

print("\n--- Evaluating Traditional CV Pipeline (with Benchmarking) ---")
for cls in classes:
    class_path = os.path.join(test_dir, cls)
    if not os.path.exists(class_path):
        continue
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(class_path, img_file)
     
            features, p_time = process_image(img_path)
            
            if features is not None:
                X_test.append(features)
                y_test.append(class_to_label[cls])
                inference_times.append(p_time)

X_test = np.array(X_test)
y_test = np.array(y_test)
avg_time = np.mean(inference_times) if inference_times else 0

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n=== SVM (Traditional CV) Results ===")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Avg Inference:   {avg_time*1000:.2f} ms per image")
print(f"Precision/F1:    {precision:.4f} / {f1:.4f}")

print("\n=== Method Comparison Summary ===")
print(f"{'Method':<20} | {'Accuracy':<10} | {'Efficiency (ms)':<15}")
print("-" * 55)
print(f"{'SVM (with GLCM)':<20} | {accuracy:<10.4f} | {avg_time*1000:<15.2f}")

if os.path.exists('cnn_final_acc.npy'):
    cnn_acc = np.load('cnn_final_acc.npy')[0]
    print(f"{'CNN (TensorFlow)':<20} | {cnn_acc:<10.4f} | {'~5-10':<15}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (SVM Accuracy: {accuracy:.2%})')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("\nConfusion matrix saved as 'confusion_matrix.png'")
