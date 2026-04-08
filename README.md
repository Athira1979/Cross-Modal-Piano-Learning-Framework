#  Spatiotemporal Attention and Cross-Modal Fusion Framework for Intelligent Piano Learning with Personalized Feedback Generation
[![MIT License](https://github.com/user-attachments/assets/f0fc5011-9ca9-485f-bfb6-1661708a0be5)](https://raw.githubusercontent.com/Athira1979/Cross-Modal-Piano-Learning-Framework/refs/heads/main/LICENSE)
 
A state-of-the-art multimodal deep learning system for automatic piano skill assessment using audio, gesture, and posture data. Achieves robust classification of beginner/intermediate/advanced skill levels.

✨ Features
Adaptive Wavelet + MFCC audio feature extraction
Multi-scale TCN + Attention for temporal modeling
Cross-modal fusion with MHGCA (Multi-Head Gated Cross-Attention)
Spectral + Spatial Attention (PASA/STABlock)
Robust preprocessing with outlier detection & normalization
End-to-end trainable with <100ms inference
📊 Architecture Overview

Copy code

🏗️ Architecture Overview
markdown

Copy code
```
Audio (Wavelet+MFCC)  → AudioEncoder 
         │
Gesture   → FDMMA (TCN+ATFM) 
         │
Posture   → STABlock (PASA+PATA) 
         │
         ↓ CrossModal Fusion (CMTPF)
         │
          → Classifier  → [Beginner/Intermediate/Advanced]
```
🚀 Quick Start
1. Install Dependencies
```bash

Copy code
 
pip install torch torchaudio pywt numpy
```

2. Prepare Dataset
```

Copy code
dataset/
├── participant1/
│   └── session1/
│       ├── audio.npy      # Raw audio (16000Hz)
│       ├── gesture.npy    # (T, J, C) or (T, features)
│       ├── posture.npy    # (T, J, C) or (T, features)
│       └── meta.json      # {"skill_level": "beginner|intermediate|advanced"}
3. Train Model
 ```
 
