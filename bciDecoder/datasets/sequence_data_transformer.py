import torch
import numpy as np
from transformers import BertModel, BertTokenizer


from datasets.utils.text_processor import (
    SIL_DEF, PHONE_DEF_SIL,
    clean, phonemezation,
    phoneToId, convert_to_ascii)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
lm = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True, )



def get_sentence_embedding(transcripts):
    lm.eval()
    tokenized_text = tokenizer(transcripts, return_tensors="pt", padding=True, truncation=True)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # tokens_tensor = tokens_tensor.to(device, non_blocking=True)
    # segments_tensors = segments_tensors.to(device, non_blocking=True)
    with torch.no_grad():
        outputs = lm(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        text_embedding = torch.mean(hidden_states[-1], dim=1)
    return text_embedding  

LOCATION_MAPPING = {'6v': {'start_ind': 0, 'end_ind': 128}, 
                '44': {'start_ind':128, 'end_ind': 256}}


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
            dat['spikePow'][0,i][:,LOCATION_MAPPING[loc]['start_ind']:LOCATION_MAPPING[loc]['end_ind']]
            ], 
            axis=1)
        features = _pad_time_dim(features, max_frame) # [np.newaxis, : ,:]
        input_features.append(features) # S * T (max_frame) * F (128 channels * 2)
    return input_features

def out_dataset_transform_ctc(sample, config):
    input_features = sample['inputFeatures']
    transcript = clean(sample['transcription'])

    # for BERT embedding generation
    processed_transcript = '[CLS] ' + transcript + ' [SEP]'

    classLabels = np.zeros([input_features.shape[0], config.dataset.n_classes]).astype(np.float32)
    newClassSignal = np.zeros([input_features.shape[0], 1]).astype(np.float32)
    phon_ids = np.zeros([config.dataset.max_seq_len]).astype(np.int32)

    transcript_phon = phonemezation(transcript)
    phon_ids[:len(transcript_phon)] = [phoneToId(p) for p in transcript_phon]

    ceMask = np.zeros([input_features.shape[0]]).astype(np.float32)
    ceMask[0:sample['frameLens']] = 1

    # paddedTranscription = np.zeros([config.dataset.max_seq_len]).astype(np.int32)
    # paddedTranscription[0:len(transcript)] = np.array(convert_to_ascii(transcript))
    feature = {'inputFeatures': input_features, 
        'seqClassIDs': phon_ids,
        'classLabelsOneHot': np.ravel(classLabels),
        'newClassSignal': np.ravel(newClassSignal),
        'ceMask': ceMask,
        'nTimeSteps': [sample['frameLens']],
        'nSeqElements': [len(transcript_phon)],
        # 'transcription': paddedTranscription,  
        'processed_transcript': processed_transcript, # for BERT embedding
        }    
    return feature

class SequenceBatch(object):
    def __init__(self, data, max_frame):
        self.input_features = torch.tensor(
            np.array([d['inputFeatures'] for d in data]))  # B * C * T * F
        self.seqClassIDs = torch.stack(
            [torch.tensor(d['seqClassIDs']) for d in data])
        self.targetLengths = torch.tensor(
            np.array([d['nSeqElements'] for d in data])).squeeze()
        # self.newClassSignal = torch.tensor(
        #     np.array([np.pad(d['newClassSignal'],
        #                 ((0, max_frame-len(d['newClassSignal']))))for d in data]))
        # self.ceMask = torch.tensor(
        #     np.array([np.pad(d['ceMask'], ((0, max_frame-len(d['ceMask']))))for d in data]))
        self.text_embedings = torch.tensor(
            np.array([get_sentence_embedding(d['processed_transcript']) for d in data]))

    # # custom memory pinning method on custom type
    def pin_memory(self):
        # self.input_features = self.input_features.pin_memory()
        return self

def seq_collate_wrapper(batch):
    max_input_frame = 600
    return SequenceBatch(batch, max_input_frame)