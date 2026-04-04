import cv2
import numpy as np

def segment_disease(enhanced_image):
    """
    Segment disease spots using Otsu thresholding.
    Returns: mask (binary), diseased (image with only spots)
    """
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if np.sum(mask == 255) > 0.8 * mask.size:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    diseased = cv2.bitwise_and(enhanced_image, enhanced_image, mask=mask)
    return mask, diseased