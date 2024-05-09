import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import utils, torch_utils

from config import init_args
import data
import models
import librosa

# Model checkpoint
checkpoint_path = '/home/dabin/video2sound/CondFoleyGen/specvqgan/onset_baseline/checkpoints/EXP_100_epoch_without_pretrained/checkpoint_ep30.pth.tar'
args = init_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = data.GreatestHitDataset(args, split='test')
test_loader = DataLoader(
    test_dataset,
    batch_size=1, # args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False)

net = models.VideoOnsetNet(pretrained=False).to(device)

# Load the trained model weights
net, _ = torch_utils.load_model(checkpoint_path, net, device=device, strict=True)
net.eval()

for step, batch in enumerate(test_loader):
    inputs = {
        'frames': batch['frames'].to(device)
    }
    
    # # Perform inference
    # with torch.no_grad():
    #     pred = net(inputs)
    
    # # Get the ground truth onsets
    # target = batch['label'].to(device)
    
    # # Convert predictions and ground truths to numpy arrays
    # pred_np = pred.cpu().numpy()
    # target_np = target.cpu().numpy()
    
    # # Visualize the predicted onsets and ground truths
    # plt.figure(figsize=(10, 4))
    # plt.plot(pred_np[0], label='Predicted Onsets')

    # # Plot ground truth onsets as column lines
    # onset_indices = np.where(target_np[0] == 1)[0]
    # plt.vlines(onset_indices, 0, 1, colors='r', linestyles='solid', label='Ground Truth Onsets')

    # plt.xlabel('Time')
    # plt.ylabel('Onset Probability')
    # plt.legend()
    # plt.title(f'Sample {step+1}')
    # plt.show()
    

    with torch.no_grad():
        pred = net(inputs)

    # Ground truth onsets
    target = batch['label'].to(device)
    
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Load the audio file
    # info = test_dataset.list_sample[batch['index'][0]].split('_')[0]
    info = test_dataset.list_sample[batch[step][0]].split('_')[0]
    video_path = os.path.join('/home/dabin/video2sound/CondFoleyGen/data', 'greatesthit', 'greatesthit-process-resized', info)
    audio_path = os.path.join(video_path, 'audio')
    audio_path = glob.glob(f"{audio_path}/*.wav")[0]
    y, sr = librosa.load(audio_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})

    # Original waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')

    # Predicted onsets and ground truth onsets
    ax2.plot(pred_np[0], label='Predicted Onsets')
    onset_indices = np.where(target_np[0] == 1)[0]
    ax2.vlines(onset_indices, 0, 1, colors='r', linestyles='solid', label='Ground Truth Onsets')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Onset Probability')
    ax2.legend()
    
    if step >= 4:
        break