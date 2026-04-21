import xml.etree.ElementTree as ET
import os

# --- Configuration ---
xml_file = '2022-12-23 Bjenberg 02_stabilized.xml'  # Path to your XML file
output_dir = '2022-12-23 Bjenberg 02_stabilized_labels/'   # Folder where the .txt files will be saved
image_width = 1920            # REPLACE with your video's pixel width
image_height = 1080           # REPLACE with your video's pixel height

# Map XML text labels to YOLO integer IDs
class_mapping = {
    'car': 0,
    
}
# ---------------------

os.makedirs(output_dir, exist_ok=True)

# Load the XML
tree = ET.parse(xml_file)
root = tree.getroot()

# Dictionary to group bounding boxes by their frame number
frame_data = {}

for track in root.findall('track'):
    label = track.get('label')
        
    class_id = class_mapping[label]

    for box in track.findall('box'):
        # CVAT uses outside="1" when an object leaves the screen. Do not extract these.
        if box.get('outside') == '1':
            continue

        frame_id = box.get('frame')
        
        # Extract CVAT absolute coordinates
        xtl = float(box.get('xtl')) # x top-left
        ytl = float(box.get('ytl')) # y top-left
        xbr = float(box.get('xbr')) # x bottom-right
        ybr = float(box.get('ybr')) # y bottom-right

        # Calculate YOLO normalized coordinates
        x_center = ((xtl + xbr) / 2) / image_width
        y_center = ((ytl + ybr) / 2) / image_height
        width = (xbr - xtl) / image_width
        height = (ybr - ytl) / image_height

        # Create the YOLO formatted string
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        if frame_id not in frame_data:
            frame_data[frame_id] = []
        frame_data[frame_id].append(yolo_line)

# Write out the .txt files
for frame_id, lines in frame_data.items():
    # Pad the frame number with zeros (e.g., frame_0080.txt) 
    # This helps keep files in alphabetical order to match your images
    filename = f"frame_{int(frame_id):05d}.txt" 
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

print(f"Successfully converted annotations and saved to '{output_dir}' folder.")
