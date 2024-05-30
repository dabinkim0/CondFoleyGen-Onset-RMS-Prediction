import pdb
import csv
import glob
import h5py
import io
import json
import librosa
import numpy as np
import os
import pickle
from PIL import Image
from PIL import ImageFilter
import random
import scipy
import soundfile as sf
import time
from tqdm import tqdm
import glob
import cv2
from scipy.signal import firwin, lfilter, ellip, filtfilt

import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms
from data.config import _C as config
# import kornia as K
import sys
sys.path.append('..')
from data import *
from utils import sound, sourcesep


def zero_phased_filter(x:np.ndarray):
    '''Zero-phased low-pass filtering'''
    b, a = ellip(4, 0.01, 120, 0.125) 
    x = filtfilt(b, a, x, method="gust")
    return x

def mu_law(rms:torch.Tensor, mu:int=255):
    '''Mu-law companding transformation'''
    # assert if all values of rms are non-negative
    assert torch.all(rms >= 0), f'All values of rms must be non-negative: {rms}'
    mu = torch.tensor(mu)
    mu_rms = torch.sign(rms) * torch.log(1 + mu * torch.abs(rms)) / torch.log(1 + mu)
    return mu_rms

def inverse_mu_law(mu_rms:torch.Tensor, mu:int=255):
    '''Inverse mu-law companding transformation'''
    assert torch.all(mu_rms >= 0), f'All values of rms must be non-negative: {mu_rms}'
    mu = torch.tensor(mu)
    rms = torch.sign(mu_rms) * (torch.exp(mu_rms * torch.log(1 + mu)) - 1) / mu
    return rms

@torch.no_grad
def get_mu_bins(mu, num_bins, rms_min):
    mu_bins = torch.linspace(mu_law(torch.tensor(rms_min)), 1, steps=num_bins)
    mu_bins = inverse_mu_law(mu_bins, mu)
    return mu_bins

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

    def __init__(self, list_file, frame_dir, audio_dir, config, split='train', max_sample=-1):
        self.split = split
        self.frame_rate = config.frame_rate
        self.duration = config.duration
        self.video_samples = config.video_samples
        self.audio_samples = config.audio_samples
        # self.mel_samples = config.mel_samples
        self.audio_len = config.audio_samples # seconds
        self.audio_sample_rate = config.audio_sample_rate
        self.rms_samples = config.rms_samples
        self.rms_nframes = config.rms_nframes
        self.rms_hop = config.rms_hop
        self.rms_discretize = config.rms_discretize
        if self.rms_discretize:
            self.rms_mu = config.rms_mu
            self.rms_num_bins = config.rms_num_bins
            self.rms_min = config.rms_min
            self.mu_bins = get_mu_bins(self.rms_mu, self.rms_num_bins, self.rms_min)
        # self.rgb_feature_dir = rgb_feature_dir
        # self.flow_feature_dir = flow_feature_dir
        # self.mel_dir = mel_dir
        self.frame_dir = frame_dir
        self.audio_dir = audio_dir

        with open(list_file, encoding='utf-8') as f:
            self.video_ids = [line.strip() for line in f]
        self.video_class = os.path.basename(list_file).split("_")[0]
        self.video_transform = transforms.Compose(
            self.generate_video_transform())

    # def get_data_pair(self, video_id):
    #     im_path = os.path.join(self.rgb_feature_dir, video_id+".pkl")
    #     # flow_path = os.path.join(self.flow_feature_dir, video_id+".pkl")
    #     # mel_path = os.path.join(self.mel_dir, video_id+"_mel.npy")
    #     audio_path = os.path.join(self.mel_dir, video_id+"_audio.npy")
    #     im = self.get_im(im_path)
    #     # flow = self.get_flow(flow_path)
    #     # mel = self.get_mel(mel_path)
    #     rms = self.get_rms(audio_path)
    #     if self.rms_discretize:
    #         with torch.no_grad():
    #             rms = discretize_rms(torch.tensor(rms.copy()), self.mu_bins)
    #         rms = rms.long() # torch.tensor(rms.copy(), dtype=torch.long)
    #     else:
    #         rms = torch.tensor(rms.copy(), dtype=torch.float32)
        
    #     # feature = np.concatenate((im, flow), 1)
    #     feature = im
    #     feature = torch.FloatTensor(feature.astype(np.float32))
    #     return (feature, rms, video_id, self.video_class)

    # def get_mel(self, filename):
    #     melspec = np.load(filename)
    #     if melspec.shape[1] < self.mel_samples:
    #         melspec_padded = np.zeros((melspec.shape[0], self.mel_samples))
    #         melspec_padded[:, 0:melspec.shape[1]] = melspec
    #     else:
    #         melspec_padded = melspec[:, 0:self.mel_samples]
    #     melspec_padded = torch.from_numpy(melspec_padded).float()
    #     return melspec_padded
    
    def get_rgb_audio_pair(self, video_id):
        # video_id = self.list_sample[index].split('_')[0]
        frame_path = os.path.join(self.frame_dir, video_id)
        audio_path = os.path.join(self.audio_dir, f"{video_id}.wav")
        
        # print("frame_path: ", frame_path)
        # print("audio_path: ", audio_path)
        
        frame_list = glob.glob(f'{frame_path}/img_*.jpg')
        frame_list.sort()
        frame_list = frame_list[0:int(self.video_samples)]
        
        imgs = self.read_image(frame_list)      # torch.Size([3, 150, 112, 112])
    
        # padding
        if imgs.shape[1] < self.video_samples:
            imgs_padded = torch.zeros((imgs.shape[0], self.video_samples, imgs.shape[2], imgs.shape[3]))
            imgs_padded[:, 0:imgs.shape[1], :, :] = imgs
        else:
            imgs_padded = imgs[:, 0:self.video_samples, :, :]
            
        if imgs_padded.shape[1] != self.video_samples:
            raise RuntimeError(f"imgs_padded length is not equal to video_samples: {imgs_padded.shape[1]} != {self.video_samples}")
            
        
        audio, audio_sample_rate = sf.read(audio_path, start=0, stop=1000, dtype='float64', always_2d=True)
        
        audio_len = int(self.duration * audio_sample_rate)
        audio, audio_rate = sf.read(audio_path, start=0, stop=audio_len, dtype='float64', always_2d=True)
        audio = audio.mean(-1)

        onsets = librosa.onset.onset_detect(y=audio, sr=audio_rate, units='time', delta=0.3)
        onsets = np.rint(onsets * self.frame_rate).astype(int)
        # onsets[onsets>29] = 29
        onsets[onsets > (self.video_samples - 1)] = self.video_samples - 1
        label = torch.zeros(len(frame_list))
        label[onsets] = 1

        batch = {
            'frames': imgs_padded,
            'label': label
        }
        return batch
        
    def read_image(self, frame_list):
        imgs = []
        convert_tensor = transforms.ToTensor()
        for img_path in frame_list:
            image = Image.open(img_path).convert('RGB')
            image = convert_tensor(image)
            imgs.append(image.unsqueeze(0))
        # (T, C, H ,W)
        # print(len(imgs)) # 151
        imgs = torch.cat(imgs, dim=0).squeeze()
        imgs = self.video_transform(imgs)
        imgs = imgs.permute(1, 0, 2, 3)
        # (C, T, H ,W)
        return imgs
    
    def generate_video_transform(self):
        resize_funct = transforms.Resize((128, 128))
        if self.split == 'train':
            crop_funct = transforms.RandomCrop(
                (112, 112))
            color_funct = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0, hue=0)
        else:
            crop_funct = transforms.CenterCrop(
                (112, 112))
            color_funct = transforms.Lambda(lambda img: img)

        vision_transform_list = [
            resize_funct,
            crop_funct,
            color_funct,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return vision_transform_list
    
    def __getitem__(self, index):
        # return self.get_data_pair(self.video_ids[index])
        return self.get_rgb_audio_pair(self.video_ids[index])

    def __len__(self):
        return len(self.video_ids)
    
    

class GreatestHitDataset(object):
    def __init__(self, args, split='train'):
        self.split = split
        if split == 'train':
            # list_sample = './data/greatesthit_train_2.00.json'
            list_sample = '/home/dabin/video2foley/CondFoleyGen/data/greatesthit_train_2.00.json'
        elif split == 'val':
            # list_sample = './data/greatesthit_valid_2.00.json'
            list_sample = '/home/dabin/video2foley/CondFoleyGen/data/greatesthit_val_2.00.json'
        elif split == 'test':
            # list_sample = './data/greatesthit_test_2.00.json'
            list_sample = '/home/dabin/video2foley/CondFoleyGen/data/greatesthit_test_2.00.json'
            
        # save args parameter
        self.repeat = args.repeat if split == 'train' else 1
        self.max_sample = args.max_sample

        self.video_transform = transforms.Compose(
            self.generate_video_transform(args))
        
        if isinstance(list_sample, str):
            with open(list_sample, "r") as f:
                self.list_sample = json.load(f)

        if self.max_sample > 0:
            self.list_sample = self.list_sample[0:self.max_sample]
            
        self.list_sample = self.list_sample * self.repeat

        random.seed(1234)
        np.random.seed(1234)
        
        num_sample = len(self.list_sample)
        self.frame_rate = 15
        self.duration = 10.0
        
        available_sample = []
        unavailable_sample = []
        
        # Remove samples that have less than 10 seconds of frames
        for sample in self.list_sample:
            frames_path = os.path.join('/home/dabin/video2foley/CondFoleyGen/data', 'greatesthit', 'greatesthit-process-resized', sample.split('_')[0], 'frames', '*.jpg')
            if len(glob.glob(frames_path)) >= self.frame_rate * self.duration:
                available_sample.append(sample)
            else:
                unavailable_sample.append(sample)
                
        self.list_sample = available_sample
        self.list_discard = unavailable_sample
        
        # self.list_sample = [sample for sample in self.list_sample 
        #                     if len(glob.glob(os.path.join('/home/dabin/video2foley/CondFoleyGen/data', 'greatesthit',    'greatesthit-process-resized', sample.split('_')[0], 'frames', '*.jpg'))) >= self.frame_rate * self.duration]
        
        print(f"Removed samples (less than {self.frame_rate * self.duration} frames) : {len(self.list_discard)}")
        
        if self.split == 'train':
            random.shuffle(self.list_sample)

        # self.class_dist = self.unbalanced_dist()
        print('Greatesthit Dataloader: # sample of {}: {}'.format(self.split, num_sample))


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index].split('_')[0]
        # video_path = os.path.join('data', 'greatesthit', 'greatesthit_processed', info)
        video_path = os.path.join('/home/dabin/video2foley/CondFoleyGen/data', 'greatesthit', 'greatesthit-process-resized', info)
        
        frame_path = os.path.join(video_path, 'frames')
        audio_path = os.path.join(video_path, 'audio')
        audio_path = glob.glob(f"{audio_path}/*.wav")
        audio_path = audio_path[0]
        # Unused, consider remove
        meta_path = os.path.join(video_path, 'hit_record.json')
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)

        audio, audio_sample_rate = sf.read(audio_path, start=0, stop=1000, dtype='float64', always_2d=True)
        
        frame_rate = self.frame_rate
        duration = self.duration
        
        frame_list = glob.glob(f'{frame_path}/*.jpg')
        frame_list.sort()
        # print(f"frame_list length: {len(frame_list)}")
        
        if len(frame_list) < duration * frame_rate:
            raise RuntimeError(f"frame_list length is less than duration * frame_rate: {len(frame_list)} < {duration * frame_rate}, video id: {info}")

        hit_time = float(self.list_sample[index].split('_')[-1]) / 22050
        if self.split == 'train':
            frame_start = hit_time * frame_rate + np.random.randint(10) - 5
            frame_start = max(frame_start, 0)
            frame_start = min(frame_start, len(frame_list) - duration * frame_rate)
            
        else:
            frame_start = hit_time * frame_rate
            frame_start = max(frame_start, 0)
            frame_start = min(frame_start, len(frame_list) - duration * frame_rate)
            
        frame_start = int(frame_start)
        frame_list = frame_list[frame_start: int(
            frame_start + np.ceil(duration * frame_rate))]
        
        # stop training if frame_list lemngth is less than duration * frame_rate
        if len(frame_list) < duration * frame_rate:
            raise RuntimeError(f"frame_list length is less than duration * frame_rate: {len(frame_list)} < {duration * frame_rate}")
        
        audio_start = int(frame_start / frame_rate * audio_sample_rate)
        audio_end = int(audio_start + duration * audio_sample_rate)

        imgs = self.read_image(frame_list)
        audio, audio_rate = sf.read(audio_path, start=audio_start, stop=audio_end, dtype='float64', always_2d=True)
        audio = audio.mean(-1)

        onsets = librosa.onset.onset_detect(y=audio, sr=audio_rate, units='time', delta=0.3)
        onsets = np.rint(onsets * frame_rate).astype(int)
        # onsets[onsets>29] = 29
        onsets[onsets > (frame_rate * duration - 1)] = frame_rate * duration - 1
        label = torch.zeros(len(frame_list))
        label[onsets] = 1

        batch = {
            'frames': imgs,
            'label': label
        }
        return batch

    def getitem_test(self, index):
        self.__getitem__(index)

    def __len__(self):
        return len(self.list_sample)


    def read_image(self, frame_list):
        imgs = []
        convert_tensor = transforms.ToTensor()
        for img_path in frame_list:
            image = Image.open(img_path).convert('RGB')
            image = convert_tensor(image)
            imgs.append(image.unsqueeze(0))
        # (T, C, H ,W)
        print("len of imgs: ", len(imgs)) # 151
        imgs = torch.cat(imgs, dim=0).squeeze()
        imgs = self.video_transform(imgs)
        imgs = imgs.permute(1, 0, 2, 3)
        # (C, T, H ,W)
        return imgs

    def generate_video_transform(self, args):
        resize_funct = transforms.Resize((128, 128))
        if self.split == 'train':
            crop_funct = transforms.RandomCrop(
                (112, 112))
            color_funct = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0, hue=0)
        else:
            crop_funct = transforms.CenterCrop(
                (112, 112))
            color_funct = transforms.Lambda(lambda img: img)

        vision_transform_list = [
            resize_funct,
            crop_funct,
            color_funct,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return vision_transform_list
