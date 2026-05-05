import cv2
import random
import os
import argparse


def extract_random_frames(video, frame_amount, image_path, label_path):
    # get video
    vidcap = cv2.VideoCapture(video)

    # get total number of frames
    totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    # frame counter
    frameCounter = 0

    # extract videoName
    vidName, _ = os.path.splitext(video)

    while frameCounter < frame_amount:
        print(frameCounter, frame_amount)
        # create random number
        randomFrameNumber=random.randint(0, int(totalFrames))
        
        # path to object frames
        path = f"{image_path}/{vidName}_frame{randomFrameNumber}"

        # check if frame exists already
        if not os.path.exists(path):
         #   continue
        #else:
            # set frame position
            vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)

            # read image
            success, image = vidcap.read()

            frameCounter += 1

            if success:
                
                cv2.imwrite(f"{image_path}/{vidName}_frame{randomFrameNumber}.jpg", image)
                open(f'{label_path}/{vidName}_frame_{randomFrameNumber}.txt', 'w').close()
            else: 
                raise Exception(f"Something went wrong with frame{randomFrameNumber}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select random frames from video and extract them if they don't exist in dataset")
    parser.add_argument("--video_dir", type=str, default="dataset", help="Directory containing video")
    parser.add_argument("--frames", type=int, help="Amount of frames to extract")
    parser.add_argument("--image_dir", type=str, default="dataset/image/train", help="Directory containing training images")
    parser.add_argument("--label_dir", type=str, default="dataset/labels/train", help="Directory containing training labels")
  
    args = parser.parse_args()
    extract_random_frames(args.video_dir, args.frames, args.image_dir, args.label_dir)

  
