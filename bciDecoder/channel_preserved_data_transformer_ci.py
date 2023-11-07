import hydra
import torch.utils.data


from datasets.neural_speech_dataset import NeuralDataset

from datasets.channel_preserved_data_transformer import (   
                                            channel_preserved_collate_wrapper,
                                           out_dataset_transform_channel_preserved,
                                           in_dataset_transform_channel_preserved,)



def _load_datasets(config):

    testset = NeuralDataset(config,
                                partition=config.dataset.test_partition,
                                input_transform=in_dataset_transform_channel_preserved,
                                transform=out_dataset_transform_channel_preserved)
    trainset = NeuralDataset(config,
                                    partition=config.dataset.train_partition,
                                    input_transform=in_dataset_transform_channel_preserved, 
                                    transform=out_dataset_transform_channel_preserved)
    
    test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=config['batch_size'], 
                                                collate_fn=channel_preserved_collate_wrapper,
                                                )
    train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=config['batch_size'], 
                                                collate_fn=channel_preserved_collate_wrapper,
                                                )

    return  train_loader, test_loader



@hydra.main(config_path='configs', config_name='config_ci')
def run_test(config):
    train_loader, test_loader = _load_datasets(config)
    print(f'total number of train trials across all sessions: {len([bid for bid, s in enumerate(train_loader)])}')
    print(f'total number of test trials across all sessions: {len([bid for bid, s in enumerate(test_loader)])}')
    first_batch = [sample.input_features for _, sample in enumerate(test_loader)][0]
    print(f'batch shape: {first_batch.shape}')
    assert(first_batch.shape==(config.batch_size, 5,config.dataset.max_frame,128))

if __name__ == "__main__":
    run_test()