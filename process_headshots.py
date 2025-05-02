import os
import re
import cv2
import logging
from datetime import datetime

# Set up logging
log_filename = datetime.now().strftime("%Y-%m-%d_script_errors.txt")
logging.basicConfig(
    filename=log_filename,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define directories
input_dir = "headshots"
output_dir = "processed_headshots"

# Create output directory if it doesn't exist
try:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
except Exception as e:
    logging.error(f"Failed to create output directory '{output_dir}': {str(e)}. Possible cause: Insufficient permissions or invalid path.")
    raise

# Regex pattern to match "FirstName.LastName" at the start of the filename
name_pattern = r"^([A-Z][a-z]+)\.([A-Z][a-zA-Z]+)\..*"

# Load the Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
try:
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        logging.error("Failed to load Haar cascade file: File is empty or corrupted at path '%s'.", cascade_path)
        raise ValueError("Haar cascade file is empty")
except Exception as e:
    logging.error(f"Failed to load Haar cascade file from '{cascade_path}': {str(e)}. Possible cause: File missing or corrupted.")
    raise

# Padding settings
padding_factor = 0.4  # 40% of face width and height
extra_padding = 5     # Additional pixels for tighter crop

# Process each file in the input directory
for filename in os.listdir(input_dir):
    try:
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            logging.error(f"Skipping non-image file: '{filename}'. Reason: File extension not in supported list (.jpg, .jpeg, .png).")
            print(f"Skipping non-image file: {filename}")
            continue

        match = re.match(name_pattern, filename)
        if not match:
            logging.error(f"Filename does not match pattern: '{filename}'. Expected format: 'FirstName.LastName.extension' with capitalized names.")
            print(f"Filename does not match pattern: {filename}")
            continue

        first_name, last_name = match.groups()
        new_name = f"{first_name}.{last_name}.jpg"
        image_path = os.path.join(input_dir, filename)

        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: '{filename}'. Possible causes: File is corrupted, not an image, or path is incorrect ('{image_path}').")
            print(f"Failed to read image: {filename}")
            continue

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
            logging.warning(f"No face detected in '{filename}'. Using whole image. Possible causes: No face present, low image quality, or face too small.")
            cropped = image
            print(f"No face detected in {filename}, using whole image.")

        original_height, original_width = cropped.shape[:2]
        new_width = 300
        new_height = int((original_height / original_width) * new_width)
        resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)

        output_path = os.path.join(output_dir, new_name)
        if cv2.imwrite(output_path, resized):
            print(f"Processed {filename} -> {new_name}")
        else:
            logging.error(f"Failed to write output image: '{new_name}' to '{output_path}'. Possible causes: Insufficient permissions, disk full, or invalid path.")
            print(f"Failed to write image: {new_name}")

    except Exception as e:
        logging.error(f"Unexpected error processing '{filename}': {str(e)}. Possible cause: Runtime issue or unhandled edge case.")
        print(f"Error processing {filename}: {str(e)}")