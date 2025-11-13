"""
data transforms for spacenet7 dataset
follows pastis24 transform pattern for compatibility
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy
import random
from utils.config_files_utils import get_params_values


def SpaceNet7_transform(model_config, is_training):
    """
    create transform pipeline for spacenet7 dataset
    
    args:
        model_config: model configuration dict
        is_training: whether this is for training (enables augmentation)
    
    returns:
        composed transform pipeline
    """
    dataset_img_res = 64  # patch size from dataloader
    input_img_res = model_config['img_res']
    ground_truths = ['labels']
    max_seq_len = model_config['max_seq_len']
    inputs_backward = get_params_values(model_config, 'inputs_backward', False)
    
    transform_list = []
    
    # convert to tensors
    transform_list.append(ToTensor())
    
    # normalize using pre-computed statistics
    transform_list.append(Normalize())
    
    # crop if needed (for different input resolutions)
    if dataset_img_res != input_img_res:
        transform_list.append(
            Crop(img_size=dataset_img_res, crop_size=input_img_res, 
                 random=is_training, ground_truths=ground_truths)
        )
    
    # add temporal encoding
    transform_list.append(TileDates(H=model_config['img_res'], W=model_config['img_res'], doy_bins=None))
    
    # standardize temporal length
    transform_list.append(CutOrPad(max_seq_len=max_seq_len, random_sample=False, from_start=True))
    
    # create unknown mask (for spacenet7, all pixels are known)
    transform_list.append(UnkMask(unk_class=255, ground_truth_target='labels'))
    
    # add backward inputs if needed (for bidirectional models)
    if inputs_backward:
        transform_list.append(AddBackwardInputs())
    
    # convert to THWC format
    transform_list.append(ToTHWC())
    
    return transforms.Compose(transform_list)


class ToTensor(object):
    """convert numpy arrays to torch tensors"""
    
    def __init__(self, label_type='binary', ground_truths=['labels']):
        self.label_type = label_type
        self.ground_truths = ground_truths
    
    def __call__(self, sample):
        tensor_sample = {}
        
        # images: [T, H, W, C] -> [T, C, H, W] torch tensor
        img = torch.tensor(sample['img']).to(torch.float32)
        tensor_sample['inputs'] = img.permute(0, 3, 1, 2)  # THWC -> TCHW
        
        # labels: [1, H, W] -> torch tensor
        tensor_sample['labels'] = torch.tensor(sample['labels'][0].astype(np.float32)).to(torch.float32).unsqueeze(-1)
        
        # temporal encoding
        tensor_sample['doy'] = torch.tensor(np.array(sample['doy'])).to(torch.float32)
        
        return tensor_sample


class Normalize(object):
    """
    normalize spectral bands using pre-computed statistics for spacenet7
    these values should be computed from the training set
    temporarily using approximate values based on uint8 range
    """
    
    def __init__(self):
        # computed from spacenet7 training set (42 aois, rgb bands)
        # shape (C, 1, 1) for broadcasting with [T, C, H, W]
        self.mean = torch.tensor([[[125.55]], [[111.56]], [[84.87]]]).to(torch.float32)
        self.std = torch.tensor([[[59.99]], [[47.75]], [[44.72]]]).to(torch.float32)
    
    def __call__(self, sample):
        # normalize: (x - mean) / std
        # inputs shape: [T, C, H, W], mean/std shape: [C, 1, 1]
        sample['inputs'] = (sample['inputs'] - self.mean) / self.std
        
        # normalize temporal encoding to [0, 1]
        sample['doy'] = sample['doy'] / 365.0001
        
        return sample


class Crop(object):
    """
    crop image to specified size
    if random=true, random crop for augmentation
    """
    
    def __init__(self, img_size, crop_size, random=False, ground_truths=['labels']):
        self.img_size = img_size
        self.crop_size = crop_size
        self.random = random
        if not random:
            self.top = int((img_size - crop_size) / 2)
            self.left = int((img_size - crop_size) / 2)
        self.ground_truths = ground_truths
    
    def __call__(self, sample):
        if self.random:
            top = torch.randint(self.img_size - self.crop_size, (1,))[0]
            left = torch.randint(self.img_size - self.crop_size, (1,))[0]
        else:
            top = self.top
            left = self.left
        
        # crop inputs: [T, C, H, W]
        sample['inputs'] = sample['inputs'][:, :, top:top + self.crop_size, left:left + self.crop_size]
        
        # crop ground truths
        for gt in self.ground_truths:
            sample[gt] = sample[gt][top:top+self.crop_size, left:left+self.crop_size]
        
        return sample


class TileDates(object):
    """
    tile temporal encoding (day of year) to image dimensions
    adds as additional channel
    """
    
    def __init__(self, H, W, doy_bins=None):
        self.H = H
        self.W = W
        self.doy_bins = doy_bins
    
    def __call__(self, sample):
        # inputs shape: [T, C, H, W]
        doy = self.repeat(sample['doy'], binned=self.doy_bins is not None)
        sample['inputs'] = torch.cat((sample['inputs'], doy), dim=1)  # concat along channel dim
        del sample['doy']
        return sample
    
    def repeat(self, tensor, binned=False):
        # tensor shape: [T]
        # output shape: [T, 1, H, W]
        T = tensor.shape[0]
        if binned:
            out = tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.H, self.W)
        else:
            out = tensor.view(T, 1, 1, 1).repeat(1, 1, self.H, self.W)
        return out


class CutOrPad(object):
    """
    standardize temporal sequence length
    pad with zeros or cut to max_seq_len
    """
    
    def __init__(self, max_seq_len, random_sample=False, from_start=False):
        self.max_seq_len = max_seq_len
        self.random_sample = random_sample
        self.from_start = from_start
        assert int(random_sample) * int(from_start) == 0, \
            "choose either random or from_start, not both"
    
    def __call__(self, sample):
        seq_len = deepcopy(sample['inputs'].shape[0])
        sample['inputs'] = self.pad_or_cut(sample['inputs'])
        
        if "inputs_backward" in sample:
            sample['inputs_backward'] = self.pad_or_cut(sample['inputs_backward'])
        
        if seq_len > self.max_seq_len:
            seq_len = self.max_seq_len
        
        sample['seq_lengths'] = seq_len
        return sample
    
    def pad_or_cut(self, tensor, dtype=torch.float32):
        seq_len = tensor.shape[0]
        diff = self.max_seq_len - seq_len
        
        if diff > 0:
            # pad with zeros
            tsize = list(tensor.shape)
            if len(tsize) == 1:
                pad_shape = [diff]
            else:
                pad_shape = [diff] + tsize[1:]
            tensor = torch.cat((tensor, torch.zeros(pad_shape, dtype=dtype)), dim=0)
        elif diff < 0:
            # cut sequence
            if self.random_sample:
                return tensor[self.random_subseq(seq_len)]
            elif self.from_start:
                start_idx = 0
            else:
                start_idx = torch.randint(seq_len - self.max_seq_len, (1,))[0]
            tensor = tensor[start_idx:start_idx+self.max_seq_len]
        
        return tensor
    
    def random_subseq(self, seq_len):
        return torch.randperm(seq_len)[:self.max_seq_len].sort()[0]


class UnkMask(object):
    """
    create mask for unknown/invalid pixels
    for spacenet7, all pixels are valid (no unknown class)
    """
    
    def __init__(self, unk_class, ground_truth_target):
        self.unk_class = unk_class
        self.ground_truth_target = ground_truth_target
    
    def __call__(self, sample):
        # all pixels are valid in spacenet7
        sample['unk_masks'] = (sample[self.ground_truth_target] != self.unk_class)
        return sample


class AddBackwardInputs(object):
    """
    add backward temporal sequence for bidirectional models
    """
    
    def __call__(self, sample):
        sample['inputs_backward'] = torch.flip(sample['inputs'], (0,))
        return sample


class ToTHWC(object):
    """
    convert from [T, C, H, W] to [T, H, W, C] format
    """
    
    def __call__(self, sample):
        sample['inputs'] = sample['inputs'].permute(0, 2, 3, 1)
        return sample


class HVFlip(object):
    """
    random horizontal and vertical flips for augmentation
    """
    
    def __init__(self, hflip_prob, vflip_prob, ground_truths=['labels']):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.ground_truths = ground_truths
    
    def __call__(self, sample):
        if random.random() < self.hflip_prob:
            sample['inputs'] = torch.flip(sample['inputs'], (2,))
            if "inputs_backward" in sample:
                sample['inputs_backward'] = torch.flip(sample['inputs_backward'], (2,))
            for gt in self.ground_truths:
                sample[gt] = torch.flip(sample[gt], (1,))
        
        if random.random() < self.vflip_prob:
            sample['inputs'] = torch.flip(sample['inputs'], (1,))
            if "inputs_backward" in sample:
                sample['inputs_backward'] = torch.flip(sample['inputs_backward'], (1,))
            for gt in self.ground_truths:
                sample[gt] = torch.flip(sample[gt], (0,))
        
        return sample