# Airway Dimension Estimation from Video Bronchoscopy Footage

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for a project practice focused on developing computer vision and GUI-based tools to estimate airway dimensions from video bronchoscopy (VB) footage, addressing the challenge of airway stent sizing in advanced lung cancer. The project investigates the use of a bronchoscope tool (e.g., biopsy forceps) as a scale reference to quantify airway structures in poor-quality VB footage, which is often degraded by blur, blood, and mucus. The tools include Python notebooks for geometric inference, a computer vision pipeline for tracheal cartilage detection, and two PyQt5-based GUI applications: the Polygon Dimension Estimator and Virtual Probe Provider.

## Project Overview

Late-stage lung cancer causes severe airway deformation, rendering traditional imaging (e.g., CT scans) ineffective for precise stent sizing. Clinicians at a collaborating hospital rely on subjective visual estimation from VB footage, which is imprecise. This project, conducted as part of a research group support effort, explores computer vision techniques to provide quantitative airway measurements. Key components include:

- **3D Experiments**: Python notebooks simulating bronchoscopy in Blender-generated tracheal models to test probe-based dimension estimation.
- **Computer Vision Pipeline**: A pipeline to detect C-shaped tracheal cartilages in VB images using OpenCV.
- **GUI Applications**:
  - **Polygon Dimension Estimator**: Allows manual annotation of airway cross-sections and dimension computation.
  - **Virtual Probe Provider**: Simulates and adjusts probe positioning for scale calibration.

Preliminary results show dimension estimates within 1 mm accuracy (mean error 0.451 mm) in controlled settings and successful cartilage detection in a real VB image. The tools are experimental but demonstrate potential for computer-assisted stent planning.

## Repository Structure

```plaintext
strade-bronchocam/
├── data/
│   ├── simsim/                          # generated data for tube experiment
│   ├── real_anon/                       # real VB footage, anonymized
│   ├── real2_anon_undist/               # undistorted
│   ├── trachea/                         # mainly for CV pipeline
│   └── output/                          # graphics used in the report
├── Simulated.ipynb                      # Python notebook with the tube experiment
├── Trachea.ipynb                        # Python notebook with the CV pipeline 
├── polygon_estimation_app.py            # polygon_estimation_app 
└── virtprobe_app.py                     # Virtual probe app