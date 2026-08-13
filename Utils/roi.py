import cv2
import numpy as np

def region_of_interest(image):
    height, width = image.shape

    # Define a triangular region
    polygon = np.array([[
        (int(width * 0.05), int(height*1.1)),          # bottom left
        (int(width * 0.9), int (height*1.1)),          # bottom right
        (int(width * 0.60), int(height * 0.65)),  # top right
        (int(width * 0.40), int(height * 0.65))   # top left
    ]], np.int32)
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image