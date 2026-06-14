# Vision 

Vision is a desktop app for assembly process monitoring.

It uses YOLO object detection, MediaPipe hand tracking, and simple process rules to help check whether each work step is done in the right order.

## What I Built

I built a Windows desktop vision system for assembly line process checking.

The app can load a YOLO model, read camera or video input, detect parts and tools, and compare the result with a configured process flow. It can warn when a worker skips a step, uses a forbidden item, or takes too long on one step.

The project also includes tools for label mapping, process editing, AOI golden sample setup, and quick YOLO training.

## Tech Stack

- Python 3.10
- PySide6
- OpenCV
- Ultralytics YOLO
- ONNX Runtime
- MediaPipe
- PyTorch and Torchvision
- Pillow
- PyInstaller
- PySerial

## Key Problems Solved

- Built a real-time UI around camera/video detection without blocking the main window
- Supported both YOLO `.pt` and `.onnx` models
- Supported normal boxes and OBB rotated boxes
- Added Chinese and English label mapping for easier process setup
- Built a multi-profile process editor for different work flows
- Added logic to detect skipped steps, forbidden items, and timeout cases
- Added hand tracking to help understand pick and touch actions
- Added AOI golden sample feature checking for final quality control
- Added optional serial alarm light output for shop-floor warning

## Demo Screenshots

Screenshots should be placed in:



```text
docs/screenshots/
```

Good screenshots to add:

- Main monitoring screen with detection boxes
- Process and safety configuration window
- Model label mapping window
- AOI golden sample setup window
- Training and annotation screen, if the data is safe to show

Do not include private factory data, customer names, product secrets, or real production records.

## Main Features

- PySide6 desktop user interface
- YOLO `.pt` and `.onnx` model support
- Normal boxes and OBB rotated boxes
- Chinese and English label mapping
- Process step editor with multiple profiles
- Skip-step and timeout warning
- Forbidden item warning
- Hand action tracking with MediaPipe
- AOI golden sample feature check
- Optional serial alarm light support
- Basic video recording and snapshot tools

## Project Structure

```text
main_tester.py       Main app window and vision thread
process_editor.py    Process step and safety rule editor
model_manager.py     Model label mapping editor
fast_trainer.py      Data collection, annotation, and quick YOLO training
logic_engine.py      Process rule checking
intent_engine.py     Hand intent and held object tracking
workflow_monitor.py  Future step monitor for skip-step warning
alarm_light.py       Serial alarm light control
configs/             Small model and process config files
docs/                Notes and analysis documents
demos/               Simple ONNX demo projects
```

## Large Files

Large files are not stored in GitHub.

This includes:

- YOLO model files: `*.pt`, `*.onnx`
- Training datasets
- Videos
- Logs
- Build output folders
- Captured images

Put model files in the local `models/` folder before running the app.

## Run

Use the project Python environment, then run:

```powershell
python main_tester.py
```



## Install Dependencies

The main Python packages are listed in:

```text
requirements.txt
```

Install missing packages inside the project environment.

## Build EXE

To build a Windows executable:

```powershell
.\build_exe.ps1
```

The build output is created under:

```text
dist/VisionCodex/
```

Build output is ignored by Git.

## Notes

This project is mainly for local industrial vision testing.

Before using it on another computer, check:

- Camera index and resolution
- Model files in `models/`
- Config files in `configs/`
- Serial alarm light port, if used
- Python or packaged EXE dependencies
