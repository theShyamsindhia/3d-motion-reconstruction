# 3D Motion Reconstruction Application

This application captures video frames and reconstructs them into 3D objects using motion tracking and Structure from Motion (SfM) techniques.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install COLMAP (optional for advanced reconstruction):
   - Download from https://colmap.github.io/
   - Add to system PATH

## Usage

1. Capture frames:
```bash
python capture_frames.py
```
This will use your default camera to capture frames. Press 'q' to stop capturing.

2. Run reconstruction:
```bash
python reconstruction.py
```
This will process the captured frames and create a 3D visualization.

## Notes

- The current implementation includes a basic reconstruction pipeline
- For better results, consider using COLMAP for dense reconstruction
- Ensure good lighting and camera movement for better 3D reconstruction
- Avoid fast movements while capturing frames

## Advanced Usage

For better reconstruction results:
1. Capture frames with ~60% overlap between consecutive frames
2. Move the camera in a smooth arc around the subject
3. Ensure the subject remains static
4. Maintain good lighting conditions

## Requirements

- Python 3.8+
- OpenCV
- Open3D
- NumPy
- Pillow
