import os
import random
import shutil

dataset_path = "mini_dataset"  
output_path = "split_data"

train_path = os.path.join(output_path, "train")
test_path = os.path.join(output_path, "test")

for folder in [train_path, test_path]:
    if not os.path.exists(folder):
        os.makedirs(folder)

for class_name in sorted(os.listdir(dataset_path)):
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    test_images = images[split_index:]
  
    os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_name), exist_ok=True)
    
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(train_path, class_name, img))
    for img in test_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(test_path, class_name, img))
    
    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

print("Data split complete!")