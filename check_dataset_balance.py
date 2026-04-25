import os
import argparse

def check_balance(image_dir, label_dir, image_ext=".jpg", label_ext=".txt"):
    images = [f for f in os.listdir(image_dir) if f.endswith(image_ext)]
    empty = 0
    annotated = 0

    for img in images:
        label = img.replace(image_ext, label_ext)
        label_path = os.path.join(label_dir, label)
        
        # Empty = no label file, or label file exists but is empty
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            empty += 1
        else:
            annotated += 1

    total = empty + annotated
 
    print(f"Total images:     {total}")
    print(f"With cars:        {annotated} ({100*annotated/total:.1f}%)")
    print(f"Without cars:     {empty} ({100*empty/total:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check balance of annotated vs empty images")
    parser.add_argument("--image_dir", type=str, default="dataset/images/train", help="Directory containing training images")
    parser.add_argument("--label_dir", type=str, default="dataset/labels/train", help="Directory containing training labels")
    parser.add_argument("--image_ext", type=str, default=".jpg", help="Extension of image files")
    parser.add_argument("--label_ext", type=str, default=".txt", help="Extension of label files")
    args = parser.parse_args()
    check_balance(args.image_dir, args.label_dir, args.image_ext, args.label_ext)