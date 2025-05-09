import os
import re
import cv2
import logging
from datetime import datetime

# Set up logging
script_name = os.path.splitext(os.path.basename(__file__))[0] if '__file__' in globals() else 'image_processing_script'
log_filename = datetime.now().strftime(f"%Y-%m-%d_{script_name}_activity.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO, # Capture info, warnings, and errors
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Also log to console for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


# Define directories
input_dir = "headshots"
output_dir = "processed_headshots"
existing_processed_dir = "existing_processed_headshots" # Directory containing already processed files

# Create output directory if it doesn't exist
try:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: '{output_dir}'")
except Exception as e:
    logging.error(f"Failed to create output directory '{output_dir}': {str(e)}. Possible cause: Insufficient permissions or invalid path.")
    raise SystemExit(f"Critical error: Cannot create output directory '{output_dir}'. Exiting.")


# Get a set of already processed filenames
existing_processed_files = set()
if os.path.exists(existing_processed_dir):
    try:
        for f_name in os.listdir(existing_processed_dir):
            # Ensure we only add actual files to the set, not subdirectories
            if os.path.isfile(os.path.join(existing_processed_dir, f_name)):
                existing_processed_files.add(f_name)
        logging.info(f"Found {len(existing_processed_files)} files in '{existing_processed_dir}' to check against.")
    except Exception as e:
        logging.warning(f"Failed to read directory '{existing_processed_dir}': {str(e)}. Will proceed without skipping based on this directory, and cleanup based on it will also be skipped.")
else:
    logging.info(f"Directory '{existing_processed_dir}' not found. No files will be skipped based on its content, and no cleanup based on it will occur.")


# Regex pattern to match "FirstName.LastName" at the start of the filename
name_pattern = r"^([A-Z][a-z]+)\.([A-Z][a-zA-Z]+)\..*"

# Load the Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    logging.error(f"Haar cascade file not found at path '{cascade_path}'. Please ensure OpenCV is correctly installed or provide a valid path.")
    raise SystemExit(f"Critical error: Haar cascade file missing at '{cascade_path}'. Exiting.")

try:
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        logging.error(f"Failed to load Haar cascade file: File is empty or corrupted at path '{cascade_path}'.")
        raise SystemExit(f"Critical error: Haar cascade file empty or corrupted at '{cascade_path}'. Exiting.")
except Exception as e:
    logging.error(f"Failed to load Haar cascade file from '{cascade_path}': {str(e)}. Possible cause: File missing or corrupted.")
    raise SystemExit(f"Critical error: Could not load Haar cascade from '{cascade_path}'. Exiting.")


# Padding settings
padding_factor = 0.4  # 40% of face width and height
extra_padding = 5     # Additional pixels for tighter crop

# Process each file in the input directory
files_processed_count = 0
files_skipped_count = 0
files_error_count = 0

if not os.path.exists(input_dir):
    logging.error(f"Input directory '{input_dir}' not found. Please create it and add images.")
    raise SystemExit(f"Critical error: Input directory '{input_dir}' not found. Exiting.")

input_file_list = []
try:
    input_file_list = os.listdir(input_dir)
    if not input_file_list:
        logging.info(f"No files found in input directory '{input_dir}'.")
except Exception as e:
    logging.error(f"Failed to list files in input directory '{input_dir}': {str(e)}")
    raise SystemExit(f"Critical error: Cannot access input directory '{input_dir}'. Exiting.")


for filename in input_file_list:
    try:
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')): # Expanded supported types
            logging.info(f"Skipping non-image file or unsupported extension: '{filename}'.")
            files_skipped_count += 1
            continue

        image_path = os.path.join(input_dir, filename)
        new_name = "" # Initialize new_name

        match = re.match(name_pattern, filename)
        if match:
            first_name, last_name = match.groups()
            base_name = f"{first_name}.{last_name}"
            original_ext = os.path.splitext(filename)[1].lower()
            # Always convert .jpeg and .png to .jpg
            if original_ext in ['.jpeg', '.png']:
                new_name = f"{base_name}.jpg"
                logging.info(f"Converting '{filename}' from {original_ext} to .jpg format.")
            else:
                # Keep original extension for other formats (including .jpg)
                new_name = f"{base_name}{original_ext}"
            logging.info(f"Filename '{filename}' matches pattern. Proposed new name: '{new_name}'.")
        else:
            logging.warning(f"Filename '{filename}' does not match 'FirstName.LastName' pattern. Will use original filename for saving.")
            # For non-matching filenames, still convert jpeg/png to jpg
            original_ext = os.path.splitext(filename)[1].lower()
            if original_ext in ['.jpeg', '.png']:
                base_name = os.path.splitext(filename)[0]
                new_name = f"{base_name}.jpg"
                logging.info(f"Converting non-pattern file '{filename}' from {original_ext} to .jpg format.")
            else:
                new_name = filename

        # --- Check if the file (by its new_name) already exists in the existing_processed_headshots directory ---
        if new_name in existing_processed_files:
            logging.info(f"OMITTING: '{filename}' (would be '{new_name}') because target name already exists in '{existing_processed_dir}'.")
            files_skipped_count += 1
            continue
        # --- End of check ---

        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: '{filename}'. Possible causes: File is corrupted, not a recognized image format, or path is incorrect ('{image_path}').")
            files_error_count += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))

        cropped = image
        if len(faces) > 0:
            if len(faces) > 1:
                logging.warning(f"Multiple faces ({len(faces)}) detected in '{filename}'. Processing the largest one.")
            
            areas = [w * h for (x, y, w, h) in faces]
            max_index = areas.index(max(areas))
            x, y, w, h = faces[max_index]
            
            pad_w = int(padding_factor * w) + extra_padding
            pad_h = int(padding_factor * h) + extra_padding
            
            left = max(0, x - pad_w)
            top = max(0, y - pad_h)
            right = min(image.shape[1], x + w + pad_w)
            bottom = min(image.shape[0], y + h + pad_h)
            
            if top < bottom and left < right:
                cropped = image[top:bottom, left:right]
                logging.info(f"Face detected and cropped for '{filename}'.")
            else:
                logging.warning(f"Invalid crop dimensions for '{filename}' after padding. Using whole image.")
        else:
            logging.warning(f"No face detected in '{filename}'. Using whole image.")

        original_height, original_width = cropped.shape[:2]
        if original_width == 0 or original_height == 0:
            logging.error(f"Cropped image for '{filename}' has zero dimension (W:{original_width}, H:{original_height}). Skipping resize and save.")
            files_error_count +=1
            continue

        new_width = 300
        new_height = int((original_height / original_width) * new_width)
        
        resized_image = cropped
        if new_height > 0 and new_width > 0:
            try:
                resized_image = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA if original_width > new_width else cv2.INTER_LINEAR)
                logging.info(f"Image '{filename}' resized to {new_width}x{new_height}.")
            except Exception as resize_err:
                logging.error(f"Error resizing image '{filename}': {str(resize_err)}. Using pre-resize cropped image.")
        else:
            logging.warning(f"Invalid resize dimensions calculated for '{filename}' (W:{new_width}, H:{new_height}). Using original cropped image dimensions.")

        output_path = os.path.join(output_dir, new_name)
        try:
            if cv2.imwrite(output_path, resized_image):
                logging.info(f"Successfully processed '{filename}' -> '{new_name}'")
                files_processed_count += 1
            else:
                logging.error(f"Failed to write output image: '{new_name}' to '{output_path}'. cv2.imwrite returned false.")
                files_error_count += 1
        except Exception as imwrite_err:
            logging.error(f"Exception during writing output image: '{new_name}' to '{output_path}': {str(imwrite_err)}.")
            files_error_count += 1

    except Exception as e:
        logging.error(f"Unexpected critical error processing '{filename}': {str(e)}. Traceback:", exc_info=True)
        files_error_count += 1

logging.info("-------------------- PROCESSING SUMMARY (PRE-CLEANUP) --------------------")
logging.info(f"Total files in input directory: {len(input_file_list)}")
logging.info(f"Successfully processed and saved to '{output_dir}': {files_processed_count}")
logging.info(f"Skipped during processing (non-image, target already in existing, etc.): {files_skipped_count}")
logging.info(f"Errors encountered during processing: {files_error_count}")

# --- Post-processing: Remove files from output_dir if they exist in existing_processed_dir ---
files_removed_from_output_count = 0
logging.info(f"--- Starting cleanup of '{output_dir}' based on contents of '{existing_processed_dir}' ---")

if os.path.exists(output_dir) and existing_processed_files: # Only proceed if output exists and we have a list of existing files
    try:
        output_files_for_cleanup = os.listdir(output_dir)
        logging.info(f"Found {len(output_files_for_cleanup)} files in '{output_dir}' to check for cleanup.")
        
        for f_name_in_output in output_files_for_cleanup:
            if f_name_in_output in existing_processed_files:
                file_to_remove_path = os.path.join(output_dir, f_name_in_output)
                try:
                    os.remove(file_to_remove_path)
                    logging.info(f"REMOVED: '{f_name_in_output}' from '{output_dir}' as it exists in '{existing_processed_dir}'.")
                    files_removed_from_output_count += 1
                except OSError as e:
                    logging.error(f"Failed to remove '{f_name_in_output}' from '{output_dir}': {str(e)}")
        
        logging.info(f"Cleanup of '{output_dir}' complete. Removed {files_removed_from_output_count} file(s).")

    except Exception as e:
        logging.error(f"An error occurred during the cleanup of '{output_dir}': {str(e)}")
elif not os.path.exists(output_dir):
    logging.info(f"Output directory '{output_dir}' does not exist. Skipping cleanup phase.")
elif not existing_processed_files:
    logging.info(f"No files to check against from '{existing_processed_dir}' (directory might be empty, non-existent, or unreadable). Skipping cleanup phase.")


logging.info("-------------------- FINAL SUMMARY --------------------")
logging.info(f"Total files in input directory: {len(input_file_list)}")
logging.info(f"Successfully processed and newly saved to '{output_dir}': {files_processed_count}")
logging.info(f"Skipped during processing (non-image, target name pre-existing, etc.): {files_skipped_count}")
if files_removed_from_output_count > 0 :
    logging.info(f"Files REMOVED from '{output_dir}' post-processing because they were also in '{existing_processed_dir}': {files_removed_from_output_count}")
logging.info(f"Net files in '{output_dir}' after this run (processed minus removed, if applicable): {files_processed_count - files_removed_from_output_count}") # This assumes files_processed_count are the only ones that could be removed by this logic.
logging.info(f"Errors encountered during processing: {files_error_count}")
logging.info(f"Log file generated at: {log_filename}")
logging.info("----------------------------------------------------------")
print(f"Script finished. Summary logged to {log_filename}")