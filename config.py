
import torch


class Config:
    # AWavelet-MFCC-TD
    FRAME_RATE = 100
    SAMPLE_RATE = 16000
    WINDOW_LENGTH = 0.0256
    MEL_FILTERS = 40
    DFT_SIZE = 512
    PRE_EMPHASIS = 0.97

    # FDMMA
    EXPANSION_COEFF = 2
    WINDOW_SIZE = 20
    HIDDEN_DEPTH = 4
    CHANNELS = 24

    # Training
    BATCH_SIZE = 32
    EPOCHS = 150
    LR = 2e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for demo
    FRAME_LEN = 0.1
    NUM_FRAMES = 20
