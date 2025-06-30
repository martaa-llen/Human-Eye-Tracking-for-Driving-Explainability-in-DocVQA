# Human-Eye-Tracking-for-Driving-Explainability-in-DocVQA

This repository contains the code for the paper "Human Eye-Tracking for Driving
Explainability in DocVQA". This project integrates two primary components: a real-time eye-tracking system and a DocVQA model, to explore how human gaze can enhance explainability in visual document analysis. 

This work is based on two projects: 
1. **Eye-Tracking**: A system for conducting eye-tracking experiments and analyzing gaze data, adapted from  [luckydeignan/Eye-Tracking](https://github.com/luckydeignan/Eye-Tracking).
2. **pDocVQA**: A probing-based DocVQA model from [andreybarsky/pdocvqa_satml](https://github.com/andreybarsky/pdocvqa_satml).

## Project Structure 

The project is organized in two main folders, each containing one of the core components of this project.

### Eye-Tracking/
This component provides a framework for conducting eye-tracking experiments using a Tobii Pro Spark eye tracker and analyzing the collected data. It is specifically designed to study human eye movements when answering questions related to the DocVQA dataset. The system runs the experiment, records gaze data, and then provides tools for analysis.

The original TraceOverview.md and README.md are also included in the folder for more detailed information.

#### Key Features & Modifications
- **Experiment GUI**: A user interface (main_GUI.py) built with Tkinter that displays document images and corresponding questions while recording gaze data from the Tobii eye tracker.
- **Data Analysis**: A comprehensive analysis script (main_data_analysis.py) that processes the stored experiment data.
- **Heatmap Generation**: Creates visual heatmaps to represent a user's gaze distribution over a document image.
- **Fixation Point Clustering**: Identifies fixation points by clustering gaze data in time and space and displays them on a scatterplot.
- **OCR Correlation**: Finds and returns the nearest OCR (Optical Character Recognition) text segments for each identified fixation point.
- **My Additions**:
  - `application.py`: A master GUI application that serves as the central control panel for the entire experimental workflow.
  - `consent_GUI.py`: A multilingual (Catalan, Spanish, English) digital consent form to manage participant onboarding and data collection.
  - `open_tobii.py`: A utility script to quickly launch the Tobii Pro Eye Tracker Manager for calibration.
  - `main_GUI.py`: The core experiment interface (adapted from the original) that displays documents and questions while recording gaze data.
  - `main_data_analysis.py`: An analysis script (adapted from the original) for processing and visualizing raw gaze data from a single participant, now with multi-layered heatmaps separating behavioral phases (pre-typing, typing, final glance).
  - `compare_humans_data.py`: A new analysis script that aggregates data from all participants to generate "human consensus heatmaps," identifying the most commonly attended regions for different answer-correctness groups.
  - `compare_with_model.py`: The final analysis script that quantitatively compares the human consensus heatmaps against the AI model's attention maps using Spearman's Rank Correlation and Top-N Overlap metrics.

#### Setup and Usage
1. Navigate to the eye-tracking directory:
```
cd Eye-Tracking
```
2. Create a Python 3.8 virtual environment. This is required for compatibility with the Tobii Pro Spark hardware.
3. Install the required dependencies in `requirements.txt`
4. *Setup Tobii Hardware*: Download and install the [Tobii Pro Eye Tracker Manager](https://connect.tobii.com/s/etm-downloads?language=en_US) and configure your Tobii Pro Spark device.
5. Configure Paths: Before running, you must modify the hyperparameter variables at the top of the Python files, especially folder_path and ocr_data in main_GUI.py.
6. Launch the Main Application:

```
python application.py
```

The GUI will provide buttons to step through the process:
  1. *Consent greement page*: Register a new participant.
  2. *Calibrate Tobii Pro Spark*: Open the Tobii software for calibration.
  3. *Trial*: Run the eye-tracking experiment for the participant.
  4. *Data Analysis*: Run the analysis script `main_data_analysis.py` for the last participant's data collected.

To run main_data_analysis.py: 
- **Standalone Usage**: This script can be run directly from the command line with the following arguments:
  - Run Mode (`--mode`):
    - `single`: Processes a single trial.
    - `full`: Processes all trials.
    - `questions`: Processes the questions appeared during the trials.
  - Analysis Mode (`--analysis-mode`):
    - default: Generates regular heatmaps.
    - `bbox_heatmaps`: Generates heatmaps based on bounding boxes.
  - Example Commands:
    ```
    # Run in single mode with default heatmap
    python main_data_analysis.py --mode single
    
    # Run in full mode with bounding box heatmaps
    python main_data_analysis.py --mode full --analysis-mode bbox_heatmaps
    ```








