#  Spatiotemporal Attention and Cross-Modal Fusion Framework for Intelligent Piano Learning with Personalized Feedback Generation
[![MIT License](https://github.com/user-attachments/assets/f0fc5011-9ca9-485f-bfb6-1661708a0be5)](https://raw.githubusercontent.com/Athira1979/Cross-Modal-Piano-Learning-Framework/refs/heads/main/LICENSE)
 
A state-of-the-art multimodal deep learning system for automatic piano skill assessment using audio, gesture, and posture data. Achieves robust classification of beginner/intermediate/advanced skill levels.
## ✨ Features
In this paper, the aim is to implement a method for enhancing piano skill development and participation. The existing methods have various drawbacks for feature extraction. This framework handles such problems by considering effective techniques. Its main contributions are summarized as follows,
+ Adaptive Wavelet-Enhanced Mel Frequency Cepstral Coefficients with Temporal Dynamics (AWavelet-MFCC-TD): This module uses integrated wavelet decomposition, MFCCs, and temporal dynamics to reduce noise.
+ Frequency-Domain Micro-Motion Analysis (FDMMA): This module uses an Adaptive Time-Frequency Temporal Convolutional Network (ATF-TCN) to accurately capture subtle finger and hand micro-motions.
+ Spatio-Temporal Attention-Driven Tracker (STAT): This module uses posture-aware temporal and spatial attention layers to track and evaluate body alignment during piano practice.
+ Cross-Modal Transformer with Modality-Specific Probing and Fusion (CMTPF): This module uses modality-specific probing to keep a balance between audio, gesture, and posture.
+ Real-Time Feedback Prioritization Network (RFP-Net): This module uses an encoder–decoder feedback system that delivers prioritized, actionable corrections on rhythm, fingering, and posture, providing adaptive guidance.
These contributions enhance piano skill development.
## 🚀 Quick Start
### 1. Clone & Install
```bash
https://github.com/Athira1979/Cross-Modal-Piano-Learning-Framework.git
cd Cross-Modal-Piano-Learning-Framework
pip install -r requirements.txt
```
### 2. Generate Dataset
```bash
python generate_dataset.py --num_sessions 1000 --split
```
### 3. Train Model
```
python train.py
```
### 4. Live Demo
```
# Webcam + Mic (default)
python demo.py live --width 1280 --height 720

# Web interface  
python demo.py web --share

# Simulation
python demo.py sim --duration 60
```
## 📊 Performance
 
<table style="width:100%">
  <tr>
    <th>Modality</th>
    <th colspan="5">Common metric</th>
    <th colspan="3">Special additional metric</th>
  </tr>
  <tr>
    <td> </td>
    <td>Accuracy (%)</td>
    <td>Precision (%)</td>
    <td>Recall (%)</td>
    <td>Specificity (%)</td>
    <td>F1-Score (%)</td>
    <td>PE (cents)</td>
    <td>JPE (mm)</td>
    <td>JAE (°)</td>
  </tr>
  <tr> 
    <td>Audio</td>
    <td>88.92</td>
    <td>87.69</td>
    <td>85.90</td>
    <td>84.99</td>
    <td>86.78</td>
    <td>12.3</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>    
    <td>Gesture</td>
    <td>86.29</td>
    <td>86.79</td>
    <td>86.12</td>
    <td>86.29</td>
    <td>86.45</td>
    <td>-</td>
    <td>15.7</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Posture</td>
    <td>89.90</td>
    <td>86.92</td>
    <td>89.56</td>
    <td>88.87</td>
    <td>88.22</td>
    <td>-</td>
    <td>-</td>
    <td>4.8</td>
  </tr>
  <tr>    
    <td>Fused model (proposed)</td>
    <td>98.58</td>
    <td>97.76</td>
    <td>97.12</td>
    <td>97.10</td>
    <td>97.43</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>
 
## 🏗️ Architecture Overview
```
                       Data Collection
                              ↓
                      Data Preprocessing
                              ↓
                      Feature Extraction
[AWavelet-MFCC-TD (Audio) + FDMMA (Gesture) + STAT (Posture)]
                              ↓
                 Cross-Modal & Fusion (CMTPF)
                              ↓
                      RFP-Net (Feedback)
                              ↓
                 Personalized Feedback Output
               [Beginner/Intermediate/Advanced]
```
## 🗂️ File Structure
```
Cross-Modal-Piano-Learning-Framework/
├── checkpoints/               # Trained models
├── data/dataset_loader.py     # PyTorch Dataset loader
├── dataset/
│   ├── audio/audio.wav        # Raw audio (16000Hz)
│   ├── gesture/gesture.npy    # Gesture features
│   ├── posture/posture.npy    # Posture features
│   └── metadata.csv           # Generated data
├── models/
│   ├── awavelet_mfcc_td.py    # AWavelet_MFCC_TD
│   ├── cmtpf.py               # CMTPF
│   ├── fdmma.py               # FDMMA
│   ├── rfp_net.py             # RFP-Net
│   └── stat.py                # STAT
├── utils/
│   ├── metrics.py             # Evaluation
│   └── plot_metrics.py        # Visualization
├── LICENSE                    # LICENSE
├── README.MD                  # Readme file
├── config.py                  # Hyperparams 
├── demo.py                    # Live/Web demos
├── evaluate.py                # Validation
├── generate_dataset.py        # Synthetic data
├── inference.py               # Inference
├── main.py                    # PianoAI Model 
├── requirements.txt           # Requirements
└── train.py                   # Training script 
 ```
## 🔧 Requirements
```
pip install torch torchvision torchaudio
pip install opencv-python mediapipe gradio pyaudio soundfile
pip install pandas numpy tqdm matplotlib seaborn scikit-learn
```
## 🎓 Training from Scratch
```
# 1. Generate data
python generate_dataset.py --num_sessions 5000 --realistic --split

# 2. Train  
python train.py

# 3. Evaluate
python metrics.py --predictions predictions.csv --plot
```
## 📄 License
<a href='https://raw.githubusercontent.com/Athira1979/Cross-Modal-Piano-Learning-Framework/refs/heads/main/LICENSE'>MIT License</a> - Free for commercial use!
