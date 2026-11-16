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

    Fashion-Image-Consistency-QA/
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
| Evaluated joint angles | 8 |

These results represent one exploratory test case.

Lower geometric differences may indicate greater similarity in the measured features, but the values are not calibrated quality scores.

## Installation

Python 3.11 is recommended.

Clone the repository and open the project directory:

    git clone https://github.com/engarashaskari-design/Fashion-Image-Consistency-QA.git
    cd Fashion-Image-Consistency-QA

Create a virtual environment:

    python3.11 -m venv .venv

Activate the environment on macOS or Linux:

    source .venv/bin/activate

Attempt installation of the recorded dependency versions (see the known compatibility issue below):

    python -m pip install -r requirements-lock.txt

Known compatibility issue: in the existing Python 3.11.0 macOS Apple Silicon environment, `python -m pip check` reports `mediapipe 0.10.21 is not supported on this platform`. The installed wheel metadata declares `cp311-cp311-macosx_14_0_x86_64`, which does not match the arm64 runtime, despite the originally installed wheel having a `macosx_11_0_universal2` filename.

`requirements-lock.txt` records pinned environment versions, not a verified portable installation. Its former temporary local wheel reference has been replaced with `mediapipe==0.10.21`; this does not fix the upstream wheel metadata. Clean-environment installation and cross-platform reproducibility have not been validated. No dependency upgrades or metadata workarounds are included.

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

The distance is the mean Euclidean distance between corresponding 2D landmarks after centering and normalization by root-mean-square face radius. It is dimensionless; rotation and perspective are not aligned.

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

Angles are computed in 2D pixel coordinates with image aspect ratio accounted for. All three landmarks in each joint must have visibility of at least 0.6 in both images; fewer than three usable angles returns `Not assessable`. The mean joint-angle difference is expressed in degrees.

## Image Generation Attribution

The fashion transformation used in the example test was generated separately using FASHN VTON v1.5.

FASHN VTON is a third-party virtual try-on model.

Official resources:

- https://github.com/fashn-AI/fashn-vton-1.5
- https://huggingface.co/spaces/fashn-ai/fashn-vton-1.5

This repository implements the image analysis, comparison logic, and Gradio interface.

It does not implement, train, or claim ownership of the FASHN image generation model.

Private source and generated example images are intentionally excluded from this repository. The recorded example results are retained from the original exploratory test and cannot be independently reproduced from the published files alone.

Third-party analysis and UI dependencies include [MediaPipe](https://github.com/google-ai-edge/mediapipe), [OpenCV](https://github.com/opencv/opencv), [NumPy](https://numpy.org/), [Pillow](https://python-pillow.github.io/), and [Gradio](https://github.com/gradio-app/gradio). Their respective licenses apply to those components.

## Limitations

- Facial geometry measurements do not verify identity preservation.
- Joint-angle similarity does not guarantee anatomical correctness.
- Body proportions are not independently validated.
- Results can be affected by image framing, occlusion, camera angle, clothing, and detection errors.
- Single-person images are expected; multi-person matching is not supported.
- No validated acceptance thresholds are provided.
- Further testing is required before production deployment.

## Development Status

Functional prototype.

The facial geometry and pose analysis components have been successfully tested with one real virtual try-on image pair.

Broader validation, automated regression testing, and clean-environment installation verification remain future work.
