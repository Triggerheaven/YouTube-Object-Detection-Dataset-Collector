# YouTube-Object-Detection-Dataset-Collector V.1
Automated GUI tool to collect images and annotations from YouTube videos using YOLO/ONNX for creating object detection datasets.
**Discord:** @xthemast

## Description

YouTube-Object-Detection-Dataset-Collector is a Python-based graphical application designed to automate the process of collecting images of specific objects (primarily configured for 'person' class, treated as 'enemy') from YouTube gameplay videos. It searches YouTube, downloads relevant videos, extracts frames, uses YOLO (v5/v8/v9) or ONNX models for object detection, crops the detected objects to a fixed size with padding, and saves the cropped images along with their corresponding annotation labels in various formats (darknet, Pascal VOC, COCO). This tool is useful for building custom datasets for training object detection models.

## Features

*   **YouTube Video Search:** Searches YouTube for videos based on user-provided keywords.
*   **Video Download:** Downloads videos using `yt-dlp` with configurable quality settings.
*   **Frame Extraction:** Extracts frames from videos at a specified time interval (highly recommends **FFmpeg** for efficiency).
*   **Object Detection:** Detects objects using Ultralytics YOLO (`.pt` models) or ONNX (`.onnx` models). Currently configured to detect the 'person' class (ID 0) by default.
*   **Custom Model Support:** Allows users to specify a path to their own custom `.pt` or `.onnx` model files.
*   **Pre-Detection Cropping:** Optionally crops frames to a specified square size around the center *before* running detection.
*   **Post-Detection Cropping:** Crops a fixed-size square region (default 416x416) centered around each detected object, adding black padding if the crop region extends beyond frame boundaries.
*   **Label Generation:** Creates annotation files for the saved cropped images in:
    *   **darknet:** `.txt` files
    *   **Pascal VOC:** `.xml` files
    *   **COCO:** A single `.json` file containing all annotations for the run.
*   **Configurable Parameters:** Allows setting search terms, max videos, frame interval, video quality, detection confidence threshold, desired number of images, label format, and cropping options via a GUI.
*   **Graphical User Interface (GUI):** Easy-to-use interface built with CustomTkinter.
*   **Background Processing:** Runs the main collection process in a separate thread to keep the GUI responsive.
*   **Concurrency:** Uses a ThreadPoolExecutor to save images and labels concurrently for potentially faster processing.
*   **Progress Monitoring:** Displays detailed progress for various stages (Search, Download, Extraction, Detection, Cropping, Saving).
*   **Export Functionality:** Allows exporting the collected images and labels into a single ZIP archive.
*   **Logging:** Records detailed information about the process into `process_log.txt`.

## Requirements

*   **Python:** Version 3.8 or higher recommended.
*   **pip:** Python package installer (usually comes with Python).
*   **FFmpeg:** **Highly Recommended** for efficient frame extraction. The application includes a check for FFmpeg and will warn if it's not found in the system PATH. While it might attempt to use OpenCV's fallback, this can be very slow or fail.
*   **Python Libraries:** Listed in the `requirements.txt` file.

## Installation

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
    Alternatively, download the `auto_dataset_v5.1.py` script directly.

2.  **Install FFmpeg:**
    This is the most crucial external dependency. You need to install it and ensure it's accessible from your system's PATH.

    *   **Windows:**
        1.  Download the latest **release build** (not git master, look for `release_essentials.zip` or similar) from the official FFmpeg website or recommended builds like those from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
        2.  Extract the downloaded ZIP archive (e.g., to `C:\ffmpeg`).
        3.  Add the `bin` directory inside the extracted folder (e.g., `C:\ffmpeg\bin`) to your system's **PATH environment variable**.
            *   Search for "Edit the system environment variables" in the Windows search bar.
            *   Click "Environment Variables...".
            *   Under "System variables" (or "User variables" for just your user), find the `Path` variable, select it, and click "Edit...".
            *   Click "New" and paste the full path to the `bin` directory (e.g., `C:\ffmpeg\bin`).
            *   Click OK on all windows.
            *   You might need to restart your command prompt or IDE for the changes to take effect.

    *   **macOS:**
        The easiest way is using Homebrew:
        ```bash
        brew install ffmpeg
        ```

    *   **Linux (Debian/Ubuntu):**
        Use the package manager:
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```

    *   **Verification:**
        Open a *new* terminal or command prompt and run:
        ```bash
        ffmpeg -version
        ```
        If it prints the FFmpeg version information, it's installed correctly and in your PATH.

3.  **Set up Python Environment (Recommended):**
    It's best practice to use a virtual environment to avoid conflicts with other Python projects.
    ```bash
    # Create a virtual environment named 'venv'
    python -m venv venv

    # Activate the virtual environment
    # On Windows (cmd/powershell):
    venv\Scripts\activate
    # On macOS/Linux (bash/zsh):
    source venv/bin/activate
    ```

4.  **Install Python Dependencies:**
    Create a file named `requirements.txt` in the same directory as the script with the following content:

    ```text
    # requirements.txt
    customtkinter
    opencv-python
    yt-dlp
    ultralytics
    Pillow
    numpy
    onnxruntime
    # Add other potential low-level dependencies if needed, but these cover the main ones.
    ```

    Install the libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `onnxruntime` is needed if you intend to use `.onnx` models. If you only use `.pt` models, you might be able to skip it, but it's included for full functionality.*

## How to Use

1.  **Launch the Application:**
    Make sure your virtual environment is activated (if you created one). Navigate to the directory containing the script and run:
    ```bash
    python auto_dataset_v5.1.py
    ```

2.  **Configure Settings:**
    The GUI provides several options in the "Einstellungen" (Settings) section:
    *   **YouTube Search Query:** Enter keywords for the videos you want to find (e.g., "warzone gameplay no commentary", "cod clips").
    *   **Max. Videos to Check:** The maximum number of videos to search for and potentially process.
    *   **Frame Interval (sec.):** The time gap between extracted frames (e.g., `5` means extract one frame every 5 seconds). Lower values yield more frames but take longer.
    *   **Video Quality (max):** Preferred maximum video resolution for download (`best`, `1080p`, `720p`, `480p`). `720p` is often a good balance.
    *   **YOLO/ONNX Model:** Select a built-in Ultralytics model or choose "Custom Model (.pt/.onnx)".
        *   If "Custom Model" is selected, a new field appears to browse for your `.pt` or `.onnx` model file.
    *   **Detection Threshold:** The minimum confidence score (0.01-1.0) for an object detection to be considered valid (e.g., `0.4` means 40% confidence).
    *   **Label Format:** Choose the output format for annotation files (`darknet`, `pascal_voc`, `coco`).
    *   **Desired Image Count:** The target number of images you want to collect. The process will stop automatically once this number of images has been successfully saved.
    *   **Pre-Detection Crop:** Check this box to enable cropping frames around their center *before* detection.
        *   **Target Size (Pixels):** If Pre-Detection Crop is enabled, specify the desired square dimension (e.g., `416`).

3.  **Start Processing:**
    Click the **"Start"** button. The process will begin in the background.

4.  **Monitor Progress:**
    *   The **Status Label** shows the current high-level task.
    *   The **Progress Bars** indicate the progress of each stage for the current video and the overall goal:
        1.  Video Search: Finding video URLs.
        2.  Video Download: Downloading the current video.
        3.  Frame Extraction: Extracting frames from the current video using FFmpeg/OpenCV.
        4.  Pre-Crop (Center): Applying the optional center crop before detection.
        5.  YOLO/ONNX Detection: Running the model on the frames.
        6.  Post-Crop / Preparation: Cropping around detections and preparing for saving.
        7.  Image/Label Saving: Shows progress of saving tasks submitted/completed (can represent submission rate during processing and completion rate afterwards).
        8.  Temp Cleanup: Cleaning up the downloaded video file.
        9.  Overall Image Goal: Progress towards the "Desired Image Count".

5.  **Stop Processing (Optional):**
    Click the **"Stop"** button (becomes active during processing) to gracefully interrupt the process. It will attempt to finish any in-progress file saves before fully stopping.

6.  **Access Output:**
    *   Collected images and their corresponding label files are saved in the `output_images` directory.
    *   If COCO format was selected, a `coco_annotations_*.json` file will be created in `output_images` at the end.
    *   Downloaded videos are temporarily stored in `temp_videos` and should be deleted automatically after processing or when the application closes.
    *   Detailed logs are written to `process_log.txt`.
    *   Click the **"Open Folder"** button to quickly open the `output_images` directory in your file explorer.

7.  **Export Results:**
    Click the **"Export as ZIP"** button to package the contents of the `output_images` directory into a single ZIP archive. You will be prompted to choose a location and filename for the archive.

## Troubleshooting

*   **Frame Extraction Failed/Slow:** Ensure **FFmpeg** is correctly installed and accessible in your system PATH. Check `process_log.txt` for specific FFmpeg errors.
*   **Model Loading Failed:** Verify the model path is correct (especially for custom models). Ensure you have the necessary libraries installed (`ultralytics`, `pytorch`, `onnxruntime` if using ONNX). Check for CUDA/GPU compatibility issues in the log.
*   **No Detections/Few Images Saved:**
    *   Lower the **Detection Threshold**.
    *   Check if the **Pre-Detection Crop** settings are appropriate (if enabled). Is the target object usually in the center?
    *   Ensure the selected model is suitable for detecting the target object ('person' class 0).
    *   Review the **YouTube Search Query** to ensure relevant videos are being found.
    *   Check `process_log.txt` for any errors during detection or saving.
*   **`yt-dlp` Errors:** Video downloads can fail due to network issues, geographic restrictions, or changes in YouTube. Check the log file for details.
*   **General Errors:** Consult the `process_log.txt` file for detailed error messages and stack traces.
