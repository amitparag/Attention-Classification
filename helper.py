import os
import cv2



def count_frames(video_path):
    frame_count = 0

    # Open the video file using PyAV
    container = av.open(video_path)

    # Iterate through the video and count frames
    for frame in container.decode(video=0):
        frame_count += 1

    return frame_count

def count_frames_in_folder(folder_path):
    # List all .avi files in the specified folder
    avi_files = [f for f in os.listdir(folder_path) if f.endswith(".avi")]

    avi_files_sorted = sorted(avi_files)

    for avi_file in avi_files_sorted:
        video_path = os.path.join(folder_path, avi_file)
        frame_count = count_frames(video_path)
        print(f"Total frames in {avi_file}: {frame_count}")

def rename_mp4_files(directory):
    mp4_files = [filename for filename in os.listdir(directory) if filename.endswith('.mp4')]
    mp4_files.sort()  # Sort the files alphabetically
    index = 1

    for mp4_file in mp4_files:
        new_name = f"{index:04d}.mp4"  # Format the new name with four-digit numbers
        old_path = os.path.join(directory, mp4_file)
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {mp4_file} -> {new_name}")
        index += 1


def rename_avi_files(folder_path):
    """
    Numbered list
    """
    
    file_extension = '.avi'
    
    # List all AVI files in the folder
    avi_files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    
    # Sort the list of AVI files
    avi_files.sort()
    
    # Rename the files in a numbered way
    for i, avi_file in enumerate(avi_files, start=1):
        old_path = os.path.join(folder_path, avi_file)
        new_name = f"{i}.avi"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    
    print("AVI files have been renamed.")






def change_resolution_recursive(root_dir, target_resolution=(240, 240)):
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.avi'):
                input_video_path = os.path.join(root, filename)
                output_video_path = os.path.join(root, filename)

                # Open the input video file
                video_reader = cv2.VideoCapture(input_video_path)

                # Get the original video's frame width and height
                original_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define the output codec and VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec (XVID is commonly used for AVI)
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, target_resolution)

                try:
                    while True:
                        ret, frame = video_reader.read()
                        if not ret:
                            break

                        # Resize the frame to the target resolution
                        resized_frame = cv2.resize(frame, target_resolution)

                        # Write the resized frame to the output video
                        video_writer.write(resized_frame)
                except Exception as e:
                    print(e)

                # Release the video capture and video writer objects
                video_reader.release()
                video_writer.release()

          
                print(f"Changed resolution of '{input_video_path}' to {target_resolution[0]}x{target_resolution[1]}")

if __name__=='__main__':
    # Example usage:
    root_directory = './dataset'  # Replace with the path to the root directory
    change_resolution_recursive(root_directory, target_resolution=(240, 240))
