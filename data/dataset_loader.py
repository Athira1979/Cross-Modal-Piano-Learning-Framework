#dataset_loader.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import soundfile as sf

class PianoDataset(Dataset):
    def __init__(self, metadata_path, max_len=5):
        self.df = pd.read_csv(metadata_path)
        self.max_len = max_len  # seconds (for fixed input)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------------------
        # AUDIO
        # -------------------
        audio, sr = sf.read(row["audio_path"])

        max_samples = sr * self.max_len
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            pad = max_samples - len(audio)
            audio = np.pad(audio, (0, pad))

        audio = torch.tensor(audio).float()

        # -------------------
        # HAND
        # -------------------
        hand = np.load(row["hand_path"])

        max_frames_hand = self.max_len * 120
        hand = hand[:max_frames_hand]

        if len(hand) < max_frames_hand:
            pad = max_frames_hand - len(hand)
            hand = np.pad(hand, ((0, pad), (0, 0), (0, 0)))

        hand = torch.tensor(hand).float()

        # -------------------
        # POSTURE
        # -------------------
        posture = np.load(row["posture_path"])  # (T, 10, 3)

        max_frames_posture = self.max_len * 30
        posture = posture[:max_frames_posture]

        if len(posture) < max_frames_posture:
            pad = max_frames_posture - len(posture)
            posture = np.pad(posture, ((0, pad), (0, 0), (0, 0)))

        # ✅ SIMPLIFIED: Just average over time
        posture = posture.mean(axis=0).reshape(1, -1)  # (1, 30)
        posture = torch.tensor(posture).float().squeeze(0)  # (30,)

        # Hand data (keep as is):
        hand = np.load(row["hand_path"])  # (T, 42, 3)
        max_frames_hand = self.max_len * 120
        hand = hand[:max_frames_hand]
        if len(hand) < max_frames_hand:
            pad = max_frames_hand - len(hand)
            hand = np.pad(hand, ((0, pad), (0, 0), (0, 0)))
        hand = torch.tensor(hand).float()  # (T, 42, 3)


        # -------------------
        # LABEL (SKILL LEVEL)
        # -------------------
        skill_map = {
            "beginner": 0,
            "intermediate": 1,
            "advanced": 2
        }

        label = skill_map[row["skill"]]
        label = torch.tensor(label).long()

        return audio, hand, posture, label
