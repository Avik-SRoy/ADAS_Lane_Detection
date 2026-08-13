import cv2
from Utils.preprocess import to_grayscale, gaussian_blur, canny_edge
from Utils.roi import region_of_interest
from Utils.lane_det import detect_lines, average_slope_intercept
from Utils.draw import draw_lane_lines, make_coordinates, fill_lane_area
from Utils.smoothing import LaneSmoother
from Utils.control import compute_lane_center, get_steering_direction


# Path to your video
video_path = "data/Sample.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

smoother = LaneSmoother()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break
    
    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    # Convert to grayscale
    gray=to_grayscale(frame)
    
    # Apply Gaussian blur
    blur=gaussian_blur(gray,5)
    
    # Apply Canny edge detection
    edges=canny_edge(blur,70,150)
    
    # Apply region of interest mask
    masked_edges=region_of_interest(edges)
    
    # Detect lines using Hough Transform
    lines=detect_lines(masked_edges)
    
    # Calculate average slope and intercept for left and right lanes
    left_lane, right_lane = average_slope_intercept(frame, lines)
    
    # Smooth the lane lines using exponential moving average
    left_lane, right_lane = smoother.smooth(left_lane, right_lane)
    
    # Draw detected lanes on the original frame
    line_image = draw_lane_lines(frame, left_lane, right_lane)
    
    # Fill the area between the lanes
    filled_frame = fill_lane_area(frame, left_lane, right_lane)
    
    # Overlay the line image on the original frame
    combined_image = cv2.addWeighted(filled_frame, 0.8, line_image, 1, 0)
    
    

    cv2.imshow("Region of Interest", masked_edges)
    cv2.imshow("Detected Lanes", combined_image)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()