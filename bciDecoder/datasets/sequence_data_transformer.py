import os
from pathlib import Path
import unicodedata
import re
import string

import torch
import numpy as np

from datasets.utils.text_processor import (
    SIL_DEF, PHONE_DEF_SIL,
    clean, phonemezation,
    phoneToId, convert_to_ascii)



LOCATION_MAPPING = {'6v': {'start_ind': 0, 'end_ind': 128}, 
                '44': {'start_ind':128, 'end_ind': 256}}

nClasses = 40
maxSeqLen = 500

def in_dataset_transform_ctc(dat,  loc, n_trials, max_frame=600):
    # TODO: truncate with a warning
    def _pad_time_dim(a, max_frame):
        if a.shape[0] > max_frame:
            return np.array(a[:max_frame, :])
        else:
            return np.pad(a, ((0,max_frame-a.shape[0]),(0,0)), 'constant')
    input_features = []
    for i in range(n_trials):    
        #get time series of TX and spike power for this trial
        features = np.concatenate(
            [dat['tx1'][0,i][:,LOCATION_MAPPING[loc]['start_ind']:LOCATION_MAPPING[loc]['end_ind']], 
            dat['spikePow'][0,i][:,LOCATION_MAPPING[loc]['start_ind']:LOCATION_MAPPING[loc]['end_ind']]], 
            axis=1)
        features = _pad_time_dim(features, max_frame)[np.newaxis, : ,:]
        input_features.append(features) # S * 1 * T (max_frame) * F (128 channels * 2)
    return input_features

def out_dataset_transform_ctc(sample):
    input_features = sample['inputFeatures']
    transcript = clean(sample['transcription'])

    classLabels = np.zeros([input_features.shape[0], nClasses]).astype(np.float32)
    newClassSignal = np.zeros([input_features.shape[0], 1]).astype(np.float32)
    phon_ids = np.zeros([maxSeqLen]).astype(np.int32)

    transcript_phon = phonemezation(transcript)
    phon_ids[:len(transcript_phon)] = [phoneToId(p) for p in transcript_phon]

    ceMask = np.zeros([input_features.shape[0]]).astype(np.float32)
    ceMask[0:sample['frameLens']] = 1

    paddedTranscription = np.zeros([maxSeqLen]).astype(np.int32)
    paddedTranscription[0:len(transcript)] = np.array(convert_to_ascii(transcript))
    feature = {'inputFeatures': input_features, 
        'seqClassIDs': phon_ids,
        'classLabelsOneHot': np.ravel(classLabels),
        'newClassSignal': np.ravel(newClassSignal),
        'ceMask': ceMask,
        'nTimeSteps': [sample['frameLens']],
        'nSeqElements': [len(transcript_phon)],
        'transcription': paddedTranscription, 
        }   
    return feature

class SequenceBatch(object):
    def __init__(self, data, max_frame):
        self.input_features = torch.tensor(
            np.array([d['inputFeatures'] for d in data]))  # B * C * T * F
        self.seqClassIDs = torch.tensor(
            np.array([d['seqClassIDs'] for d in data]))
        # self.newClassSignal = torch.tensor(
        #     np.array([np.pad(d['newClassSignal'],
        #                 ((0, max_frame-len(d['newClassSignal']))))for d in data]))
        # self.ceMask = torch.tensor(
        #     np.array([np.pad(d['ceMask'], ((0, max_frame-len(d['ceMask']))))for d in data]))
        self.transcription = torch.tensor(
            np.array([d['transcription'] for d in data]))

    # # custom memory pinning method on custom type
    def pin_memory(self):
        # self.input_features = self.input_features.pin_memory()
        return self

def seq_collate_wrapper(batch):
    max_input_frame = 600
    return SequenceBatch(batch, max_input_frame)