import cv2
import matplotlib.pyplot as plt
import os
from enhancement_step import enhance_image
from segmentation_step import segment_disease

def generate_report_figure(image_path, save_path='processing_pipeline.png'):
    """
    Generates a 1st-class visualization of the CV pipeline for the report.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    enhanced = enhance_image(img)
    mask, result = segment_disease(enhanced)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(enhanced_rgb)
    plt.title("2. Enhanced (CLAHE)")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("3. Segmentation Mask")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(result_rgb)
    plt.title("4. Final Segmented Spot")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Report figure saved as '{save_path}'")

if __name__ == "__main__":
    sample_dir = "split_data/test/Apple___Apple_scab"
    if os.path.exists(sample_dir):
        sample_img = os.path.join(sample_dir, os.listdir(sample_dir)[0])
        generate_report_figure(sample_img)
    else:
        print("Please ensure 'split_data' exists before running visuals.")
