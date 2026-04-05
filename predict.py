import torch
import joblib
import cv2
import numpy as np
import os
from pipeline import process_image
from model_cnn import create_cnn_model 

CLASSES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]
IMG_SIZE = (128, 128)

def predict_single_image(image_path):
    print(f"\nAnalyzing Image: {os.path.basename(image_path)}")
    print("-" * 30)

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    if os.path.exists('plant_disease_svm.joblib'):
        svm = joblib.load('plant_disease_svm.joblib')
        features, _ = process_image(image_path)
        if features is not None:
            svm_pred_idx = svm.predict([features])[0]
            print(f"SVM Prediction:   {CLASSES[svm_pred_idx]}")
        else:
            print("SVM Prediction:   Failed (Could not extract features)")
    else:
        print("SVM Prediction:   Unavailable (Run train_evaluate.py first)")

    if os.path.exists('plant_disease_cnn.pth'):
        num_classes = len(CLASSES)
        model = create_cnn_model(num_classes)
        model.load_state_dict(torch.load('plant_disease_cnn.pth'))
        model.eval()

        img_resized = cv2.resize(img, IMG_SIZE)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, cnn_pred_idx = torch.max(outputs, 1)
            print(f"CNN Prediction:   {CLASSES[cnn_pred_idx.item()]}")
    else:
        print("CNN Prediction:   Unavailable (Run train_cnn.py first)")

if __name__ == "__main__":
    test_dir = "split_data/test/Apple___Black_rot"
    if os.path.exists(test_dir):
        sample_img = os.path.join(test_dir, os.listdir(test_dir)[0])
        predict_single_image(sample_img)
    else:
        print("No test folder found. Please provide an image path in the script.")
