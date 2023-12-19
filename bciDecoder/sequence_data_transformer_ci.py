import hydra
import torch.utils.data


from datasets.neural_speech_dataset import NeuralDataset

from datasets.sequence_data_transformer import (   
                                            seq_collate_wrapper,
                                           in_dataset_transform_ctc,
                                           out_dataset_transform_ctc,)



def _load_datasets(config):

    testset = NeuralDataset(config,
                                partition=config.dataset.test_partition,
                                input_transform=in_dataset_transform_ctc,
                                transform=out_dataset_transform_ctc)
    trainset = NeuralDataset(config,
                                    partition=config.dataset.train_partition,
                                    input_transform=in_dataset_transform_ctc, 
                                    transform=out_dataset_transform_ctc)
    
    test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=config['batch_size'], 
                                                collate_fn=seq_collate_wrapper,
                                                )
    train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=config['batch_size'], 
                                                collate_fn=seq_collate_wrapper,
                                                )

    return  train_loader, test_loader



@hydra.main(config_path='configs', config_name='config_ci')
def run_test(config):
    train_loader, test_loader = _load_datasets(config)
    print(f'total number of train trials across all sessions: {len([bid for bid, s in enumerate(train_loader)])}')
    print(f'total number of test trials across all sessions: {len([bid for bid, s in enumerate(test_loader)])}')
    first_batch_x = [sample.input_features for _, sample in enumerate(test_loader)][0]
    print(f'feature batch shape: {first_batch_x.shape}')
    assert(first_batch_x.shape==(config.batch_size, 1,config.dataset.max_frame,256))

    first_batch_y = [sample.seqClassIDs for _, sample in enumerate(test_loader)][0]
    print(f'output batch shape: {first_batch_y.shape}')
    assert(first_batch_y.shape==(config.batch_size, config.dataset.max_seq_len))

if __name__ == "__main__":
    run_test()