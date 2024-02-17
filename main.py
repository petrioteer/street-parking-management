import cv2

car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

def detect_cars(frame, prev_frame_cars):
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (prev_x, prev_y, _, _) in prev_frame_cars:
            if abs(prev_x - x) < 2 and abs(prev_y - y) < 2:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return frame, cars

video_capture = cv2.VideoCapture('prop_video.mp4')

prev_frame_cars = []

while True:
 
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    frame_with_cars, current_frame_cars = detect_cars(frame, prev_frame_cars)
    
    prev_frame_cars = current_frame_cars
    
    cv2.imshow('Car Detection', frame_with_cars)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
