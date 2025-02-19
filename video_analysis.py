import cv2
from ultralytics import YOLO

# Load the model
yolo = YOLO('models/yolov8s.pt')

# Function to get class colors
def get_colours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def analyze_video(video_path, output_path):
    video_cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        results = yolo.track(frame, stream=True)

        for result in results:
            # get the class names
            classes_names = result.names

            # iterate over each box
            for box in result.boxes:
                # check if confidence is greater than 40 percent
                if box.conf[0] > 0.4:
                    # get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # get the class
                    cls = int(box.cls[0])

                    # get the respective colour
                    colour = get_colours(cls)

                    # draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    # put the class name and confidence on the image
                    cv2.putText(frame, f'{classes_names[cls]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        
        # write the frame to the output file
        cv2.imshow('frame', frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()