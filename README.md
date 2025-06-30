# Human-Eye-Tracking-for-Driving-Explainability-in-DocVQA

This repository contains the code for the paper "Human Eye-Tracking for Driving Explainability in DocVQA". 

This project addresses the lack of transparency in DocVQA models by comparing AI-generated attention with human cognitive processes captured via eye-tracking. We recorded gaze data from 30 participants and performed causal occlusion experiments on a multimodal model to test its reliance on human-attended information. The findings expose a critical "explainability gap," validating the use of human cognitive data as a ground truth for developing more transparent and trustworthy AI systems.

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
5. Configure Paths: Before running, you must modify the hyperparameter variables
   - `main_GUI.py`
     - `folder_path`
     - `npy_filtered_landscape_path`
   - `main_data_analysis.py`
     - `filtered_data_npy_path`
     - `PARTICIPANTS_EYE_DATA_DIR`
7. Launch the Main Application:

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
  - Save Plots for default mode (`--save-plots`)
    - If included when running in default mode the plots will be saved
    - The `bbox_heatmaps` saves the plots by default
  - Example Commands:
    ```
    # Run in single mode with default heatmap
    python main_data_analysis.py --mode single
    
    # Run in full mode with bounding box heatmaps
    python main_data_analysis.py --mode full --analysis-mode bbox_heatmaps
    
    #Run in full single, default mode and saving plots
    python main_data_analysis.py --mode single --save-plots
    ```

### pdocvqa_satml/

This component uses the VT5 model to generate predictions and attention maps for the documents used in the human trials. It was also used to run the causal occlusion experiments.

#### Key Features & Modifications
- **Textual Attention Map Generation**: The model was used to extract decoder cross-attention scores, which were aggregated to a word-level to produce an AI attention map for each document-question pair.
- **Causal Occlusion Experiments**: The core of the AI-side analysis. The model was run on modified inputs where information was selectively hidden or revealed based on the human consensus heatmaps to test for sufficiency and necessity.
- **My Additions**:
  - Generated visual and textual attention heatmaps of the model's performance.
  - Implemented the logic for the occlusion experiments (generating masked images and filtered text inputs).
  - Modified the `eval.py` to save the model's textual attention maps in a format compatible with the human data analysis pipeline.

#### Setup and Usage
1. Navigate to the pdocvqa_satml directory:
  ```
  cd pdocvqa_satml
  ```
2. Set up the Conda Environment:
  ```
  conda env create -f environment.yml
  conda activate pdocvqa_satml
  ```
3. Download the Dataset: Obtain the dataset from the ELSA Benchmarks Platform (or use the specific subset from this project).
4. Configure Paths: Modify the paths in
   - `configs/datasets/DocVQA.yml` to point to your imdb_dir and images_dir.
   - `AGGREGATED_HUMAN_JSON_DIR` in `find_human_consensus_file()` function in `eval.py` file

6. Run Training or Evaluation:
  - To train the model:
    ```
    python train.py --dataset PFL-DocVQA-BLUE --model VT5

    ```
  - To evaluate the model and generate attention maps:
    ```
    python eval.py --dataset DocVQA --model VT5
    ```


## Acknowledgments
A special thank you to the original authors of the Eye-Tracking and pdocvqa_satml repositories for their foundational work.

I am deeply grateful to all 30 participants who generously volunteered their time for this study.

My sincere appreciation to my tutor, Andrey Barsky, and the Computer Vision Center (CVC) for their guidance and for providing the Tobii Pro Spark eye-tracker.



