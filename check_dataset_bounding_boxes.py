import cv2

video_path = '' # add video path here
label_path = '' # add corresponding label path here

def parse_yolo_to_pixels(yolo_line, img_w, img_h):
    """Converts YOLO format (class x_center y_center w h) to (x_min, y_min, x_max, y_max)"""
    parts = yolo_line.strip().split()
    x_center, y_center, w, h = map(float, parts[1:5])
    
    # Calculate pixel coordinates
    x_min = int((x_center - w/2) * img_w)
    y_min = int((y_center - h/2) * img_h)
    x_max = int((x_center + w/2) * img_w)
    y_max = int((y_center + h/2) * img_h)
    
    return x_min, y_min, x_max, y_max


def bound_box_video(video_path, label_path):
    """Takes in video path, and label_path for said video. Draws rectangle around labeled area on each frame."""
    # Open input video
    print("test")
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output writer (mp4v codec for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
        
        file_path = f"{label_path}/frame_{frame_count:05d}.txt"
        print(frame_count)
        try:
            with open(file_path, "r") as f:
                content = f.read().splitlines()
                print(content)
                # Draw rectangle if current frame is in target list
                for box_string in content:
                    x1, y1, x2, y2 = parse_yolo_to_pixels(box_string, width, height)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Save frame to new video
            out.write(frame)
            print(frame_count)
            frame_count += 1
        except FileNotFoundError:
            out.write(frame)
            print(frame_count)
            frame_count += 1

        
    # Cleanup
    cap.release()
    out.release()



bound_box_video(video_path=video_path,label_path=label_path)