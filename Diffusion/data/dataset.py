import torch 
from torch import nn 
from functools import partial
import numpy as np 
from Diffusion.data.base import Txt2ImgIterableBaseDataset
from torch.utils.data import DataLoader, Dataset
from Diffusion.utils import instantiate_from_config

def worker_init_fn(_):

    # get worker information from pytorch
    worker_info = torch.utils.data.get_worker_info()

    # Access the dataset assigned to this worker
    dataset = worker_info.dataset
    # Get current worker ID (0, 1, 2, ..., num_workers-1)
    worker_id = worker_info.id 

    # check for specific dataset type 
    if isinstance(dataset, Txt2ImgIterableBaseDataset):

        # calculation equal split size per worker.
        split_size = dataset.num_records // worker_info.num_workers

        # Assign subset of data to this worker
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size: (worker_id + 1) * split_size]

        # Get random state arrary length (typically 624 for NumPy)
        state_arr = np.random.get_state()[1]

        # Randomly select an index from the state array 
        current_id = np.random.choice(len(state_arr), 1)

        # seed = state array value at random position + worker ID
        return np.random.seed(state_arr[current_id] + worker_id)
    
    else:

        # For other dataset types 
        # seed = first value in state array + worker ID
        return np.random.seed(np.random.get_state()[1][0] + worker_id)
    

class WrappedDataset(Dataset):

    def __init__(self, dataset):
        self.data = dataset 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    



class DataModuleFromConfig(nn.Module):

    def __init__(self,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        

        super().__init__()
        self.batch_size = batch_size
        self.dataset_config = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2 
        self.use_worker_init_fn = use_worker_init_fn

        if train is not None:
            self.dataset_config["train"] = train
            self.train_dataloader = self._train_dataloader 


        if validation is not None:
            self.dataset_config["validation"] = validation
            self.val_dataset = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)


        if test is not None:
            self.dataset_config["test"] = test 
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)

        if predict is not None:
            self.dataset_config["predict"] = predict
            self.predict_dataloader = self._predict_dataloader 

        self.wrap = wrap 

        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_config[k]))
            for k in self.dataset_config
        )


    def setup(self, stage=None):

        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_config[k]))
            for k in self.dataset_config
        )

        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])


    


    def _train_dataloader(self):

        print("check the dataset are comming or not")

        is_iterable_dataset = isinstance(self.datasets["train"],
                                         Txt2ImgIterableBaseDataset)
        
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn

        else:
            init_fn = None 

        return DataLoader(dataset=self.datasets["train"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)
    


    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets["validation"], 
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            
            init_fn = worker_init_fn

        else:
            init_fn = None 


        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)
    


    def _test_dataloader(self, shuffle=False):

        is_iterable_dataset = isinstance(self.dataset["test"], 
                                         Txt2ImgIterableBaseDataset)
        
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn

        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset 
        shuffle = shuffle and (not is_iterable_dataset)


        return DataLoader(self.dataset["test"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)
    


    def _predict_dataloader(self, shuffle=False):

        if isinstance(self.datasets['predict'], 
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:

            init_fn = worker_init_fn

        else:
            init_fn = None


        return DataLoader(self.datasets["predict"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn)
    





    








if __name__ == "__main__":

    from Diffusion.data.lsun import LSUNBedroomsTrain, LSUNBedroomsValidation

    train_config = {
        "target": "Diffusion.data.lsun.LSUNBedroomsTrain"
    }

    datasets = LSUNBedroomsTrain()

    data_moduler = DataModuleFromConfig(batch_size=32,
                                   train=train_config,
                                   num_workers=4,
                                   use_worker_init_fn=True,
                                   )

    data_loader = data_moduler.train_dataloader()
  
    for batch in data_loader:
        print(f"check the shape of image: {batch['image'].shape}")