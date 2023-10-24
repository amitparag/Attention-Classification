"""
Script to augment the dataset by adding noise, flipping channels,   

"""
import os
import cv2
import numpy as np



def transform(input_video_path:str, 
            output_video_path:str,
            noise_intensity:int=25,
            add_noise:bool=True,
            flip_frame:bool=True):
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)
    fourcc = int(cap.get(6))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Swap the Red and Blue channels
        transformed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if flip_frame:
            # Flip the frame horizontally
            transformed_frame = cv2.flip(transformed_frame, 1)  # 1 for horizontal flip, 0 for vertical flip, -1 for both



        # Add noise to the frame
        if add_noise:
            noise = np.random.normal(0, noise_intensity, transformed_frame.shape).astype(np.uint8)
            transformed_frame = np.clip(transformed_frame + noise, 0, 255)

        # Write the frame to the output video
        out.write(transformed_frame)

    # Release the video objects
    cap.release()
    out.release()

    return

def get_videos_info(directory, extensions):
    videos_info = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                videos_info.append({
                    'name': file,
                    'path': os.path.join(root, file),
                    'directory': root
                })

    return videos_info


def transform_videos(videos_info):
    
    
    for video in videos_info:

        name = video['name']
        path = video['path']
        dir  = video['directory']
        
        print("Processing:", path)

                
        transformed_video_path = dir + f'/aug_{name}'
                
        transform(input_video_path=path,output_video_path=transformed_video_path)
        print("=" * 40)
    

    print("Complete")


if __name__=='__main__':

    videos_info = get_videos_info('./dataset/learning/wriggle', 'avi')
    #transform_videos(videos_info)
