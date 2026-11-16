# Fashion Image Consistency QA

A lightweight Python computer vision application for evaluating facial geometry and body pose consistency between original and AI-generated fashion images.

## Project Overview

This project provides an interactive quality analysis workflow for fashion image transformation and virtual try-on results.

It compares two images of a person and reports exploratory geometric measurements to help identify potential visual inconsistencies.

The application performs local image analysis and does not generate fashion images itself.

## Features

- Facial landmark detection using MediaPipe FaceMesh
- OpenCV-based face detection fallback
- Normalized facial geometry comparison
- Body pose estimation using MediaPipe Pose
- Eight joint-angle comparisons
- Interactive Gradio web interface
- Structured JSON reports
- CPU-based processing without paid APIs

## Technology Stack

- Python 3.11
- MediaPipe
- OpenCV
- NumPy
- Pillow
- Gradio

## Project Structure

    Fashion_Identity_QA/
    |-- app.py
    |-- face_analysis.py
    |-- face_comparison.py
    |-- pose_analysis.py
    |-- requirements-lock.txt
    |-- README.md
    |-- .gitignore

## Example Results

The application was tested using an original person image and a styled image generated separately with FASHN VTON v1.5.

| Metric | Observed Result |
|---|---|
| Facial geometry distance | 0.0158 |
| Mean joint-angle difference | 1.32 degrees |
| Valid joint angles | 8 |

These results represent one exploratory test case.

Lower geometric differences may indicate greater similarity in the measured features, but the values are not calibrated quality scores.

## Installation

Python 3.11 is recommended.

Clone the repository and open the project directory.

Create a virtual environment:

    python3.11 -m venv .venv

Activate the environment on macOS or Linux:

    source .venv/bin/activate

Install the recorded dependencies:

    python -m pip install -r requirements-lock.txt

Important: The existing macOS Apple Silicon environment reports a MediaPipe 0.10.21 wheel compatibility warning.

Clean-environment installation and cross-platform reproducibility have not yet been validated.

## Running the Application

Start the Gradio application:

    python app.py

Open the local URL displayed in the terminal.

The application normally uses:

    http://127.0.0.1:7860

## Usage

1. Upload the original person image.
2. Upload the AI-generated or styled person image.
3. Run the image consistency analysis.
4. Review the facial geometry comparison.
5. Review the body joint-angle differences.
6. Inspect the structured JSON results.

## Analysis Methods

### Facial Geometry Analysis

Facial landmarks are extracted and compared using normalized geometric coordinates.

The reported distance provides an exploratory measure of facial landmark similarity.

It does not establish or verify personal identity.

### Body Pose Analysis

MediaPipe Pose is used to estimate body landmarks.

The application calculates angular differences for up to eight joints:

- Left elbow
- Right elbow
- Left shoulder
- Right shoulder
- Left hip
- Right hip
- Left knee
- Right knee

Only sufficiently visible joints are included in the reported comparison.

The mean joint-angle difference is expressed in degrees.

## Image Generation Attribution

The fashion transformation used in the example test was generated separately using FASHN VTON v1.5.

FASHN VTON is a third-party virtual try-on model.

Official resources:

- https://github.com/fashn-AI/fashn-vton-1.5
- https://huggingface.co/spaces/fashn-ai/fashn-vton-1.5

This repository implements the image analysis, comparison logic, and Gradio interface.

It does not implement, train, or claim ownership of the FASHN image generation model.

## Limitations

- Facial geometry measurements do not verify identity preservation.
- Joint-angle similarity does not guarantee anatomical correctness.
- Body proportions are not independently validated.
- Results can be affected by image framing, occlusion, camera angle, clothing, and detection errors.
- No validated acceptance thresholds are provided.
- Further testing is required before production deployment.

## Development Status

Functional prototype.

The facial geometry and pose analysis components have been successfully tested with one real virtual try-on image pair.

Broader validation, automated regression testing, and clean-environment installation verification remain future work.
