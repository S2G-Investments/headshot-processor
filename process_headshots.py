import os
import re
import cv2

# Define directories
input_dir = "headshots"
output_dir = "processed_headshots"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Regex pattern to match "FirstName.LastName" at the start of the filename
name_pattern = r"^([A-Z][a-z]+)\.([A-Z][a-z]+)\..*"

# Load the Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Padding settings
padding_factor = 0.2  # 20% of face width and height
extra_padding = 5     # Additional pixels for tighter crop

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        match = re.match(name_pattern, filename)
        if match:
            first_name, last_name = match.groups()
            new_name = f"{first_name}.{last_name}.jpg"
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    areas = [w * h for (x, y, w, h) in faces]
                    max_index = areas.index(max(areas))
                    x, y, w, h = faces[max_index]
                    pad_w = int(padding_factor * w) + extra_padding
                    pad_h = int(padding_factor * h) + extra_padding
                    left = max(0, x - pad_w)
                    top = max(0, y - pad_h)
                    right = min(image.shape[1], x + w + pad_w)
                    bottom = min(image.shape[0], y + h + pad_h)
                    cropped = image[top:bottom, left:right]
                else:
                    cropped = image
                    print(f"No face detected in {filename}, using whole image.")
                original_height, original_width = cropped.shape[:2]
                new_width = 300
                new_height = int((original_height / original_width) * new_width)
                resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
                output_path = os.path.join(output_dir, new_name)
                cv2.imwrite(output_path, resized)
                print(f"Processed {filename} -> {new_name}")
            else:
                print(f"Failed to read image: {filename}")
        else:
            print(f"Filename does not match pattern: {filename}")
    else:
        print(f"Skipping non-image file: {filename}")