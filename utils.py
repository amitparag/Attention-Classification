import os
import cv2
import numpy as np

def remove_black_borders(video, output_path, target_width, target_height):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (target_width, target_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply an adaptive threshold to detect black borders
        _, thresh = cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)

        # Crop the frame to the bounding box
        cropped_frame = frame[y:y+h,x:x+w]

        # Resize the frame to the target width and height
        resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

        out.write(resized_frame)  # Write the resized frame to the output video

    video.release()
    out.release()

def process_slip_videos(slip_folder):

    file_list = os.listdir(slip_folder)

    # Filter for video files with a specific extension, e.g., .avi
    video_files = [f for f in file_list if f.lower().endswith(('.avi'))]


    output_folder = slip_folder + '/no_borders' 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Iterate through video files and get their paths and names
    for video_file in video_files:
        video_path = os.path.join(slip_folder, video_file)
        print("Video Path:", video_path)
        print("Video File Name:", video_file)


        target_width = 320  # Set the target width
        target_height = 240  # Set the target height

        output_path = os.path.join(output_folder, video_file)


        cap = cv2.VideoCapture(video_path)
        remove_black_borders(cap, output_path, target_width, target_height)


def check_resolution(video_path, target_width, target_height):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None  # Video file could not be opened

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Check if the resolution matches the target
    if width == target_width and height == target_height:
        return True  # Resolution is as expected
    else:
        return False  # Resolution is different

def process_videos_in_folder(folder_path, target_width, target_height):
    # List all files in the folder, including subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.avi', '.mp4', '.mkv')):
                video_path = os.path.join(root, file)
                result = check_resolution(video_path, target_width, target_height)
                if result is None:
                    print(f"Error: Could not open video file: {video_path}")
                elif result:
                    print(f"Video resolution is correct: {video_path}")
                else:
                    print(f"Video resolution is not 320x240: {video_path}")


if __name__=='__main__':




