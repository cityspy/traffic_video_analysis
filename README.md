# Traffic Video Analysis

Python-based traffic video analysis project for extracting vehicle counts from roadway footage and exporting count summaries by frame and by minute.

## Overview

This project processes traffic videos and generates structured traffic count outputs for downstream analysis. The current repository includes:

- `main.py` — main analysis script
- `vehicle_counts_per_frame.csv` — vehicle count output by frame
- `vehicle_counts_per_minute.csv` — aggregated vehicle count output by minute

## Features

- Video-based traffic analysis workflow
- Frame-level vehicle count export
- Minute-level aggregated count export
- Python implementation suitable for further extension
- GitHub-ready project structure

## Project Structure

```text
traffic_video_analysis/
├── .gitignore
├── main.py
├── vehicle_counts_per_frame.csv
├── vehicle_counts_per_minute.csv
└── README.md
```

## Requirements

Typical Python libraries for this kind of workflow may include:

- Python 3.10+
- OpenCV
- NumPy
- Pandas
- Ultralytics YOLO (if object detection is used)

Install dependencies with:

```bash
pip install opencv-python numpy pandas ultralytics
```

## Usage

Run the main script from the project root:

```bash
python main.py
```

Depending on the script configuration, the program may:

1. Load a traffic video
2. Detect and/or track vehicles
3. Count vehicles within defined regions
4. Export results to CSV files

## Outputs

### `vehicle_counts_per_frame.csv`
Contains frame-by-frame vehicle count results.

### `vehicle_counts_per_minute.csv`
Contains aggregated vehicle counts grouped by minute.

## Customization

This project can be extended to support:

- Multiple vehicle classes
- Lane-based counting
- Region-of-interest polygon tuning
- Speed estimation
- Turning movement counts
- Intersection conflict analysis
- Visualization overlays on output video

## Notes

- Large video files should generally not be committed directly to GitHub because GitHub enforces a 100 MB file size limit on regular Git pushes.[cite:33]
- For large assets such as videos or model weights, use Git LFS or external storage links when needed.[cite:33]

## Future Improvements

- Add a `requirements.txt`
- Add sample input/output data
- Add configuration file support
- Add performance benchmarking
- Add README images or workflow diagrams
- Add result visualization dashboards

## Author

Cody Ma
