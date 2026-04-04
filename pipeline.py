import cv2
import numpy as np
import time
from enhancement_step import enhance_image
from segmentation_step import segment_disease
from feature_extraction import extract_features

def process_image(image_path):
    """
    Complete computer vision pipeline: Read -> Enhance -> Segment -> Extract.
    Returns: features (list), process_time (seconds)
    """
    start_time = time.time()
    
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0

    enhanced = enhance_image(img)

    mask, diseased = segment_disease(enhanced)

    features = extract_features(enhanced, mask)
    
    end_time = time.time()
    process_time = end_time - start_time
    
    return features, process_time
