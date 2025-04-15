# Headshot Processor

This project provides a Python script to process headshot images. It renames files based on a specific pattern, crops images to focus on the individual's face, resizes them to a width of 300 pixels, and saves them as JPEG files. The project uses [UV](https://github.com/astral-sh/uv) for project management and [OpenCV](https://opencv.org/) for image processing.

## Features

- **Rename**: Extracts "FirstName.LastName" from the filename using regex and renames the file accordingly
- **Crop**: Detects the face in the image and crops it with a 20% padding around the face. If no face is detected, the entire image is used
- **Resize**: Resizes the cropped image to a width of 300 pixels while maintaining the aspect ratio
- **Convert**: Saves the processed image as a JPEG file, converting from other formats if necessary

## Prerequisites

- **UV**: Ensure UV is installed on your system. Install it with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Python**: The script requires Python, but UV manages the virtual environment for you

## Setup

1. Clone the Repository (if hosted on a platform like GitHub):
   ```bash
   git clone https://github.com/yourusername/headshot-processor.git
   cd headshot-processor
   ```

2. Install Dependencies:
   The project uses UV to manage dependencies, which are listed in `pyproject.toml`. Install them with:
   ```bash
   uv sync
   ```
   This command sets up a virtual environment and installs required packages, including `opencv-python`.

## Usage

1. **Prepare Input Directory**: Place your headshot images in the `headshots` directory. Supported formats include JPEG and PNG.

2. **Run the Script**: Execute the script using UV to process all images in the `headshots` directory:
   ```bash
   uv run process_headshots.py
   ```
   Processed images will be saved in the `processed_headshots` directory, which is created automatically.

## Filename Format

- **Input Filename**: Filenames should start with "FirstName.LastName" followed by optional text. Example:
  ```
  Dinsh.Guzdar.headshot-4d52e4fb89c43498e1.jpg
  ```
- **Output Filename**: The processed file is renamed to "FirstName.LastName.jpg". Example:
  ```
  Dinsh.Guzdar.jpg
  ```

## Face Detection

- The script uses OpenCV's Haar cascade classifier for face detection
- If multiple faces are detected, it selects the largest one by area
- If no face is detected, the entire image is used, and a console message is displayed

## Project Structure

```
headshot-processor/
├── headshots/                # Directory for input images
├── processed_headshots/      # Directory for output images (created automatically)
├── process_headshots.py      # The main script
├── pyproject.toml           # UV project configuration
└── README.md                # This file
```

## Examples

### Example 1: Processing a Single Headshot

**Input**: Place `John.Doe.photo.png` in the `headshots` directory.

**Command**:
```bash
uv run process_headshots.py
```

**Output**: `processed_headshots/John.Doe.jpg` (cropped, resized to 300px width, saved as JPEG)

### Example 2: Handling Multiple Files

**Input**: Place `Alice.Smith.headshot.jpg` and `Bob.Jones.profile.png` in `headshots`.

**Command**:
```bash
uv run process_headshots.py
```

**Output**:
- `processed_headshots/Alice.Smith.jpg`
- `processed_headshots/Bob.Jones.jpg`

## Troubleshooting and Notes

- **Multiple Faces**: The script selects the largest face if multiple are detected. Modify face detection parameters in the script if needed.
- **No Face Detected**: If no face is found, the full image is used. Adjust the script's fallback behavior as necessary.
- **Padding**: The crop includes 20% padding around the face. Edit `pad_w` and `pad_h` in the script to adjust this.
- **Adding Dependencies**: To install additional packages, use:
  ```bash
  uv add <package-name>
  ```

```md
headshot-processor/
├── headshots/                # Directory for input images
├── processed_headshots/      # Directory for output images (created automatically)
├── process_headshots.py      # The main script
├── pyproject.toml            # UV project configuration
└── README.md                 # This file
```


Examples
Example 1: Processing a Single Headshot
Input: Place John.Doe.photo.png in the headshots directory.
Command:
sh

```bash
uv run process_headshots.py
```

Output: processed_headshots/John.Doe.jpg (cropped, resized to 300px width, saved as JPEG).

Example 2: Handling Multiple Files
Input: Place Alice.Smith.headshot.jpg and Bob.Jones.profile.png in headshots.
Command:
```bash
uv run process_headshots.py
```

Output:
processed_headshots/Alice.Smith.jpg
processed_headshots/Bob.Jones.jpg
