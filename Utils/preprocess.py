import cv2

def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def gaussian_blur(frame, kernel_size=5):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def canny_edge(frame, low_threshold=50, high_threshold=150):
    return cv2.Canny(frame, low_threshold, high_threshold)