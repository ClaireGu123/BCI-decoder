import os

import scipy.io
import numpy as np
from torch.utils.data import IterableDataset

from datasets.utils.get_speech_session_blocks import get_speech_session_blocks


def _block_wise_normalization(dat, input_features):
    # normalize within block, each session could contain multiple blocks
    # normalize channel by channel
    # input features: S * C * T * F
    # output features: S * C * T * F
    blockNums = np.squeeze(dat['blockIdx'])
    blockList = np.unique(blockNums)
    blocks = []
    for b in range(len(blockList)):
        sentIdx = np.argwhere(blockNums==blockList[b])
        sentIdx = sentIdx[:,0].astype(np.int32)
        blocks.append(sentIdx)
    input_features = np.array(input_features)
    
    for b in range(len(blocks)):
        for channel in range(input_features[0].shape[0]):
            # average within block over the time dimension
            feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1), channel], axis=0)
            feats_mean = np.mean(feats, axis=0, keepdims=True)
            feats_std = np.std(feats, axis=0, keepdims=True)
            for i in blocks[b]:
                input_features[i, channel] = (input_features[i, channel] - feats_mean) / (feats_std + 1e-8)
    return input_features

class NeuralDataset(IterableDataset):
    def __init__(self,config,partition, input_transform, transform):

        
        base_dir = config.dataset.base_dir
        self.loc = config.dataset.area
        self.max_frame = config.dataset.max_frame
        self.max_seq_len = config.dataset.max_seq_len
        self.config = config
        self.partition = partition
        self.block_lists = get_speech_session_blocks()
        self.data_path= os.path.join(base_dir, 'competitionData')
        self.input_transform = input_transform
        self.transform = transform


    def __iter__(self):
        for sess_ind in range(len(self.block_lists)):
            session_name = self.block_lists[sess_ind][0]
            mat_file = os.path.join(self.data_path, self.partition, session_name+'.mat' )
            dat = scipy.io.loadmat(mat_file)
            input_features = []
            n_trials = dat['sentenceText'].shape[0]
            #collect area 6v tx1 and spikePow features
            input_features = self.input_transform(dat, self.loc, n_trials, self.max_frame) #S * T * F (128 channels * 2)
            input_features = _block_wise_normalization(dat, input_features)
            n_trials = 1
            for i in range(n_trials):    
                sentence_len = input_features[i].shape[0]
                sentence = dat['sentenceText'][i].strip()
                yield self.transform({'inputFeatures': input_features[i],
                    'transcription': sentence,
                    'frameLens': sentence_len,
                    },
                    config = self.config,)


