import os
from yacs.config import CfgNode as CN
import librosa
import numpy as np

_C  =  CN()

_C.log = CN()
_C.log.save_dir = 'ckpt/GH_240309_GLS'
_C.log.exclude_dirs = ['ckpt', 'data', 'opencv-python']
_C.log.logger = ['tensorboard', 'wandb']
_C.log.loss = CN()
_C.log.loss.types = ["CE", "MSE", "CE_GLS"]
if "CE_GLS" in _C.log.loss.types:
    _C.log.loss.gls_num_classes = 16
    _C.log.loss.gls_blur_range = 3

_C.train = CN()
_C.train.epochs = 1000
_C.train.num_epoch_save = 50
_C.train.seed = 123
_C.train.dynamic_loss_scaling = True
_C.train.dist_backend = "nccl"
_C.train.dist_url = "tcp://localhost:54321"
_C.train.cudnn_enabled = True
_C.train.cudnn_benchmark = False
_C.train.checkpoint_path = ''
_C.train.epoch_count = 0

_C.train.loss = CN()
_C.train.loss.type = "CE_GLS"               # "CE_GLS"
if _C.train.loss.type == "CE_GLS":
    _C.train.loss.gls_num_classes = 16
    _C.train.loss.gls_blur_range = 3        
_C.train.grad_clip_thresh = 1.0
_C.train.batch_size = 512 # 64
_C.train.lr = 1e-3
_C.train.beta1 = 0.5
_C.train.continue_train = False
_C.train.lambda_Oriloss = 10000.0
_C.train.lambda_Silenceloss = 0
_C.train.weight_decay = 0.5
_C.train.niter = 100
_C.train.loss_threshold = 0.0001
_C.train.D_interval = 1
_C.train.wo_G_GAN = False
_C.train.wavenet_path = ""


_C.data = CN()
materials = os.listdir('/mnt/GreatestHits/features')
            # ['carpet', 'ceramic', 'cloth', 'dirt', 'drywall', 'glass', 'grass', 'gravel', 'leaf', 'metal', 
            #  'multiple', 'None', 'paper', 'plastic', 'plastic-bag', 'rock', 'tile', 'water', 'wood']
materials.sort()
_C.data.training_files = [f'/mnt/GreatestHits/filelists/{material}_train.txt' for material in materials]
_C.data.test_files = [f'/mnt/GreatestHits/filelists/{material}_test.txt' for material in materials]

# features for bn-inception
# _C.data.rgb_feature_dirs = [f"/mnt/GreatestHits/features/{material}/feature_rgb_bninception_dim1024_30fps" 
#                             for material in materials]
_C.data.flow_feature_dirs = [f"/mnt/GreatestHits/features/{material}/feature_flow_bninception_dim1024_30fps" 
                             for material in materials]

# features for r2plus1d
_C.data.rgb_feature_dirs = [f"/mnt/GreatestHits/features/{material}/feature_rgb_r2plus1d_dim512_30fps" 
                            for material in materials]
_C.data.frame_dirs = [f"/mnt/GreatestHits/features/{material}/OF_10s_15fps" 
                            for material in materials]
_C.data.audio_dirs = [f"/mnt/GreatestHits/features/{material}/audio_10s_16000hz" 
                      for material in materials]


_C.data.mel_dirs = [f"/mnt/GreatestHits/features/{material}/melspec_10s_16000hz" for material in materials]
_C.data.frame_rate = 15
_C.data.duration = 10
_C.data.video_samples = _C.data.frame_rate * _C.data.duration
_C.data.audio_samples = 10
_C.data.audio_sample_rate = 16000
_C.data.mel_samples = 860
_C.data.rms_nframes = 512
_C.data.n_mel_channels = 128
_C.data.rms_hop = 128
dummy_audio =np.pad(np.zeros(_C.data.audio_samples*_C.data.audio_sample_rate),
                    (int((_C.data.rms_nframes - _C.data.rms_hop) / 2), int((_C.data.rms_nframes - _C.data.rms_hop) / 2), ), 
                    mode="reflect")
_C.data.rms_samples = int(librosa.feature.rms(y=dummy_audio, \
                                            frame_length=_C.data.rms_nframes, hop_length=_C.data.rms_hop, \
                                            center=False, pad_mode="reflect").shape[1])
_C.data.rms_discretize = True
if _C.data.rms_discretize:
    _C.data.rms_mu = 255
    _C.data.rms_num_bins = 16
    _C.data.rms_min = 0.01


_C.model = CN()
_C.model.visual_dim = 512                   # bn-inception : 2048 (rgb 1024 + flow 1024)

# Encoder parameters
_C.model.random_z_dim = 512                 # bn-inception : 512
_C.model.encoder_n_lstm = 2
_C.model.encoder_embedding_dim = 512        # bn-inception : 2048 (rgb 1024 + flow 1024)
_C.model.encoder_kernel_size = 5
_C.model.encoder_n_convolutions = 3

# Auxiliary parameters
# _C.model.auxiliary_type = "lstm"
# _C.model.auxiliary_dim = 2 # 128
# _C.auxiliary_sample_rate = 32
# _C.model.mode_input = "vis"
# _C.model.aux_zero = False

# Decoder parameters
_C.model.decoder_conv_dim = 1024

# Mel-post processing network parameters
_C.model.postnet_embedding_dim = 512
_C.model.postnet_kernel_size = 5
_C.model.postnet_n_convolutions = 5

