import cv2
import numpy as np

# Load car cascade classifier (use a more advanced model if necessary)
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Optional: Load background subtractor for improved foreground detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

def stationary_detection(frame, car_cascade, fgbg=None):
    """
    Detects cars in a given frame and marks stationary ones with red boxes.

    Args:
        frame: The video frame as a NumPy array.
        car_cascade: The Haar cascade classifier for car detection.
        fgbg: Optional background subtractor for foreground detection.

    Returns:
        The frame with red boxes drawn around stationary cars.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if fgbg is not None:
        fgmask = fgbg.apply(gray)
        gray = fgmask

    # Apply car cascade classifier to detect cars
    cars = car_cascade.detectMultiScale(gray, 1.1, 4)

    # Track car positions over frames for stationary detection
    car_positions = {}

    # Iterate through detected cars
    for (x, y, w, h) in cars:
        car_key = (x, y)  # Use unique key for tracking

        # Calculate car center and speed based on previous position
        if car_key in car_positions:
            prev_x, prev_y = car_positions[car_key]
            center_x = (x + x + w) // 2
            center_y = (y + y + h) // 2
            dx, dy = abs(center_x - prev_x), abs(center_y - prev_y)
            speed = np.sqrt(dx**2 + dy**2)  # Euclidean distance as speed measure
        else:
            speed = 0.0  # No previous position, assume non-stationary initially

        # Update car position in tracker
        car_positions[car_key] = (x, y)

        # Check if car is stationary based on a speed threshold
        if speed < 5.0:  # Adjust threshold as needed
            # Draw red box around stationary car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame

# Load the video
cap = cv2.VideoCapture('sample.mp4')

# Process each frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect cars and mark stationary ones
    frame = stationary_detection(frame, car_cascade, fgbg)

    # Display the frame
    cv2.imshow('Car Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
