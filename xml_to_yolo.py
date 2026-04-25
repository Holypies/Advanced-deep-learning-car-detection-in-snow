import xml.etree.ElementTree as ET
import os
import argparse 

def cvat_xml_to_yolo(xml_path, output_label_dir, img_width=1920, img_height=1080):
    """
    Convert CVAT XML annotations to YOLO format .txt label files.
    One .txt file per frame, named filename_frame_XXXXXX.txt to match your extracted images.
    """
    os.makedirs(output_label_dir, exist_ok=True)
    xml_name = xml_path.split("/")[-1].split(".")[0].replace(" ", "_")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Collect all boxes grouped by frame number
    frame_annotations = {}  # {frame_id: [yolo_line, ...]}

    for track in root.iter("track"):
        label = track.attrib["label"]  # "car"
        class_id = 0  # only one class; adjust if you have more

        for box in track.iter("box"):
            # Skip boxes where the car is outside/not visible
            if box.attrib.get("outside") == "1":
                continue

            frame = int(box.attrib["frame"])
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            # Clamp to image boundaries (sometimes annotations slightly exceed edges)
            xtl = max(0, min(xtl, img_width))
            ytl = max(0, min(ytl, img_height))
            xbr = max(0, min(xbr, img_width))
            ybr = max(0, min(ybr, img_height))

            # Convert to YOLO format: cx, cy, w, h — all normalized 0-1
            cx = ((xtl + xbr) / 2) / img_width
            cy = ((ytl + ybr) / 2) / img_height
            w  = (xbr - xtl) / img_width
            h  = (ybr - ytl) / img_height

            yolo_line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

            if frame not in frame_annotations:
                frame_annotations[frame] = []
            frame_annotations[frame].append(yolo_line)

    # Write one .txt file per frame
    for frame_id, lines in frame_annotations.items():
        label_filename = f"{xml_name}_frame_{frame_id:06d}.txt"
        label_path = os.path.join(output_label_dir, label_filename)
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Done! Wrote labels for {len(frame_annotations)} frames → {output_label_dir}")

if __name__ == "__main__":
    # set up arguments
    parser = argparse.ArgumentParser(description="Convert CVAT XML annotations to YOLO format")
    parser.add_argument("--xml_path", required=True, help="Path to the CVAT XML annotation file")
    parser.add_argument("--output_label_dir", required=True, help="Directory to save the YOLO label files")
    parser.add_argument("--img_width", type=int, default=1920, help="Image width")
    parser.add_argument("--img_height", type=int, default=1080, help="Image height")

    args = parser.parse_args()

    cvat_xml_to_yolo(
        xml_path=args.xml_path,
        output_label_dir=args.output_label_dir,
        img_width=args.img_width,
        img_height=args.img_height
    )