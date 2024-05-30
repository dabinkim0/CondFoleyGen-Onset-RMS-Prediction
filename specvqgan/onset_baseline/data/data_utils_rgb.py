import os
import pickle
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
import librosa
from scipy.signal import firwin, lfilter, ellip, filtfilt


def zero_phased_filter(x:np.ndarray):
    '''Zero-phased low-pass filtering'''
    b, a = ellip(4, 0.01, 120, 0.125) 
    x = filtfilt(b, a, x, method="gust")
    return x

# def mu_law(rms:np.ndarray, mu:int=255):
#     '''Mu-law companding transformation'''
#     assert np.all(rms >= 0)
#     mu_rms = np.sign(rms) * np.log(1 + mu * np.abs(rms)) / np.log(1 + mu)
#     return mu_rms
def mu_law(rms:torch.Tensor, mu:int=255):
    '''Mu-law companding transformation'''
    # assert if all values of rms are non-negative
    assert torch.all(rms >= 0), f'All values of rms must be non-negative: {rms}'
    mu = torch.tensor(mu)
    mu_rms = torch.sign(rms) * torch.log(1 + mu * torch.abs(rms)) / torch.log(1 + mu)
    return mu_rms

# def inverse_mu_law(mu_rms:np.ndarray, mu:int=255):
#     '''Inverse mu-law companding transformation'''
#     assert np.all(mu_rms >= 0)
#     rms = np.sign(mu_rms) * (np.exp(mu_rms * np.log(1 + mu)) - 1) / mu
#     return rms
def inverse_mu_law(mu_rms:torch.Tensor, mu:int=255):
    '''Inverse mu-law companding transformation'''
    assert torch.all(mu_rms >= 0), f'All values of rms must be non-negative: {mu_rms}'
    mu = torch.tensor(mu)
    rms = torch.sign(mu_rms) * (torch.exp(mu_rms * torch.log(1 + mu)) - 1) / mu
    return rms

# def get_mu_bins(mu, num_bins, rms_min):
#     mu_bins = np.linspace(mu_law(rms_min), 1, num=num_bins)
#     mu_bins = inverse_mu_law(mu_bins, mu)
#     return mu_bins
@torch.no_grad
def get_mu_bins(mu, num_bins, rms_min):
    mu_bins = torch.linspace(mu_law(torch.tensor(rms_min)), 1, steps=num_bins)
    mu_bins = inverse_mu_law(mu_bins, mu)
    return mu_bins
    
# def discretize_rms(rms, mu_bins):
#     rms = np.maximum(rms, 0.0) # change negative values to zero
#     rms_inds = np.digitize(rms, mu_bins) # discretize
#     return rms_inds
def discretize_rms(rms, mu_bins):
    rms = torch.maximum(rms, torch.tensor(0.0)) # change negative values to zero
    rms_inds = torch.bucketize(rms, mu_bins, right=True) # discretize
    return rms_inds

def undiscretize_rms(rms_inds, mu_bins, ignore_min=True):
    if ignore_min and mu_bins[0] > 0.0:
        mu_bins[0] = 0.0
    
    rms_inds_is_cuda = rms_inds.is_cuda
    if rms_inds_is_cuda:
        device = rms_inds.device
        rms_inds = rms_inds.detach().cpu()
    rms = mu_bins[rms_inds]
    if rms_inds_is_cuda:
        rms = rms.to(device)
    return rms


class VideoAudioDataset(torch.utils.data.Dataset):
    """
    loads image, flow feature, audio files
    """

    def __init__(self, list_file, rgb_feature_dir, flow_feature_dir, mel_dir, config, max_sample=-1):
        self.video_samples = config.video_samples
        self.audio_samples = config.audio_samples
        # self.mel_samples = config.mel_samples
        self.audio_len = config.audio_samples # seconds
        self.audio_sample_rate = 22050
        self.rms_samples = config.rms_samples
        self.rms_nframes = config.rms_nframes
        self.rms_hop = config.rms_hop
        self.rms_discretize = config.rms_discretize
        if self.rms_discretize:
            self.rms_mu = config.rms_mu
            self.rms_num_bins = config.rms_num_bins
            self.rms_min = config.rms_min
            self.mu_bins = get_mu_bins(self.rms_mu, self.rms_num_bins, self.rms_min)
        self.rgb_feature_dir = rgb_feature_dir
        self.flow_feature_dir = flow_feature_dir
        self.mel_dir = mel_dir

        with open(list_file, encoding='utf-8') as f:
            self.video_ids = [line.strip() for line in f]
        self.video_class = os.path.basename(list_file).split("_")[0]

    def get_data_pair(self, video_id):
        im_path = os.path.join(self.rgb_feature_dir, video_id+".pkl")
        # flow_path = os.path.join(self.flow_feature_dir, video_id+".pkl")
        # mel_path = os.path.join(self.mel_dir, video_id+"_mel.npy")
        audio_path = os.path.join(self.mel_dir, video_id+"_audio.npy")
        im = self.get_im(im_path)
        # flow = self.get_flow(flow_path)
        # mel = self.get_mel(mel_path)
        rms = self.get_rms(audio_path)
        if self.rms_discretize:
            with torch.no_grad():
                rms = discretize_rms(torch.tensor(rms.copy()), self.mu_bins)
            rms = rms.long() # torch.tensor(rms.copy(), dtype=torch.long)
        else:
            rms = torch.tensor(rms.copy(), dtype=torch.float32)
        
        # feature = np.concatenate((im, flow), 1)
        feature = im
        feature = torch.FloatTensor(feature.astype(np.float32))
        return (feature, rms, video_id, self.video_class)

    # def get_mel(self, filename):
    #     melspec = np.load(filename)
    #     if melspec.shape[1] < self.mel_samples:
    #         melspec_padded = np.zeros((melspec.shape[0], self.mel_samples))
    #         melspec_padded[:, 0:melspec.shape[1]] = melspec
    #     else:
    #         melspec_padded = melspec[:, 0:self.mel_samples]
    #     melspec_padded = torch.from_numpy(melspec_padded).float()
    #     return melspec_padded
    
    def get_frames(self, frames_path):
        PASS

    def get_im(self, im_path):
        with open(im_path, 'rb') as f:
            im = pickle.load(f, encoding='bytes')
        f.close()
        if im.shape[0] < self.video_samples:
            im_padded = np.zeros((self.video_samples, im.shape[1]))
            im_padded[0:im.shape[0], :] = im
        else:
            im_padded = im[0:self.video_samples, :]
        assert im_padded.shape[0] == self.video_samples
        return im_padded

    def get_flow(self, flow_path):
        with open(flow_path, 'rb') as f:
            flow = pickle.load(f, encoding='bytes')
        f.close()
        if flow.shape[0] < self.video_samples:
            flow_padded = np.zeros((self.video_samples, flow.shape[1]))
            flow_padded[0:flow.shape[0], :] = flow
        else:
            flow_padded = flow[0:self.video_samples, :]
        return flow_padded
    
    def get_rms(self, audio_path):
        signal = np.load(audio_path)
        signal = np.pad(signal , (int((self.rms_nframes - self.rms_hop) / 2), int((self.rms_nframes - self.rms_hop) / 2), ), 
                        mode="reflect")
        rms = librosa.feature.rms(y=signal, frame_length=self.rms_nframes, hop_length=self.rms_hop, 
                                  center=False, pad_mode="reflect")
        rms = rms[0]
        if not len(rms) == self.rms_samples:
            raise RuntimeError(f"Error while calculating RMS, got length {len(rms)} instead of {self.rms_samples}.") 
        rms = zero_phased_filter(rms)
        return rms
    
    def __getitem__(self, index):
        return self.get_data_pair(self.video_ids[index])

    def __len__(self):
        return len(self.video_ids)