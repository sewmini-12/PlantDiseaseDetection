import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(enhanced_image, mask):
    """
    Extract features from the segmented disease region.
    Returns a list: [mean_R, mean_G, mean_B, area, contrast, correlation, energy, homogeneity]
    """
    if mask is None or np.sum(mask) == 0:
         return [0, 0, 0, 0, 0, 0, 0, 0]

    # 1. Mean RGB of the masked region
    mean_rgb = cv2.mean(enhanced_image, mask=mask)[:3]  # (B, G, R) order from OpenCV
    
    # 2. Area (number of white pixels in mask)
    area = np.sum(mask == 255)
    
    # 3. GLCM features (Texture) - working on grayscale
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    
    # Compute GLCM on the whole image (but mask it if possible)
    # skimage's glcm doesn't support mask natively, so we ensure background is 0
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Compute GLCM
    glcm = graycomatrix(masked_gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Return features (convert BGR to RGB order for consistency)
    return [
        mean_rgb[2], mean_rgb[1], mean_rgb[0], 
        float(area), 
        float(contrast), float(correlation), float(energy), float(homogeneity)
    ]