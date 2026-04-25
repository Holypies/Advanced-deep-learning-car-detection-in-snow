import cv2
import os
import argparse

def extract_frames(video_path, output_dir, frame_skip):
    os.makedirs(output_dir, exist_ok=True)
    video_name = video_path.split("/")[-1].split(".")[0].replace(" ", "_")
    # Load the video in a video capture obj.
    cap = cv2.VideoCapture(video_path)

    # Check if video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_idx = 0
    saved = 0

    while cap.isOpened(): # while the video is open
        ret, frame = cap.read() # ret: bool -> frame read successfully, frame: the actual frame 
        if not ret: # video has ended
            break
        if frame_idx % frame_skip == 0: # save one frame every frame_skip frames
            filename = f"{output_dir}/{video_name}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(filename, frame) 
            saved += 1
        frame_idx += 1

    cap.release() # release the video capture obj. 
    print(f"Saved {saved} frames from {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video", help="Path to the video file")
    parser.add_argument("--output", help="Output directory for extracted frames")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip interval")

    args = parser.parse_args()

    if not args.video or not args.output:
        print("Usage: python extract_frames.py --video <video.mp4> --output <output_dir> [--skip <interval>]")
        exit(1)

    extract_frames(args.video, args.output, args.skip)