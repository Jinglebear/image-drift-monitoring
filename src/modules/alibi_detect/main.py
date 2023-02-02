
# my class
from untrained_encoder import UntrainedAutoencoder

# wilds import
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds import get_dataset

# torchvision transforms
import torchvision.transforms as transforms

# misc
import numpy as np
import os
import json
def main():
    CAMELYON_UAE_DETECTOR_PATH =  '/home/jinglewsl/evoila/projects/image-drift-monitoring/src/modules/alibi_detect/uae_ks_detector_camelyon'
    CAMELYON_ROOT_PATH = '{}/data/camelyon17_v1.0'.format(os.getcwd())

    my_uae = UntrainedAutoencoder()

    """ WILDS CAMELYON DATASET """
    dataset = get_dataset(dataset="camelyon17", download=False)

    img_size = (96,96)
    """ GET SPLITS & TRANSFORM DATA"""
    # Get the training set (in distribution)
    train_data = dataset.get_subset("train",transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()]))

    # Get the validation set (in distribution)
    val_data = dataset.get_subset("val",transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()]))

    # Get the test set (out of distribution)
    # test_data = dataset.get_subset("test",transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()]))


    """ GET DATALOADER  & CREATE NUMPY DATASET WITH SIZE N (Standard Data Loader shuffles examples WILDS API)"""
    N = 5000 # Size of Dataset
    train_loader= get_train_loader('standard',train_data,1)
    train_ds = np.stack([tensor[0][0].numpy() for tensor ,i in zip(train_loader,range(N)) if i <N],axis=0)

    val_loader = get_train_loader('standard',val_data,1)
    val_ds = np.stack([tensor[0][0].numpy() for tensor ,i in zip(val_loader,range(N)) if i <N],axis=0)

   

    # # encoder_fn = my_uae.init_encoder(encoding_dim=96,input_shape=train_ds.shape[1:],batch_size=512)

    # # my_uae.init_detector(reference_data=train_ds,encoder_fn=encoder_fn, p_val= 0.05, path=CAMELYON_UAE_DETECTOR_PATH, save_dec=True)
    
    my_uae.import_detector(path=CAMELYON_UAE_DETECTOR_PATH)




    res_val_ds = my_uae.make_prediction(target_data=val_ds)

    print('H_0"\n\n')

    res_train_ds = my_uae.make_prediction(target_data=train_ds)

    # print('res_val:', res_val_ds)
    # print('res_train:', res_train_ds)
    
    dumped_val = json.dumps(res_val_ds, cls=NumpyEncoder)
    dumped_train = json.dumps(res_train_ds, cls=NumpyEncoder)
    
    with open("res_val_ds.json","w")as outfile:
        json.dump(dumped_val,outfile)
    with open("res_train_ds.json","w")as outfile:
        json.dump(dumped_train,outfile)
    

   

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# ======================================================================================
# call
if __name__ == "__main__":
    main()