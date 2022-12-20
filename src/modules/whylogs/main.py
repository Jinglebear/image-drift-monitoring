
def main():
    from whylogs_logger import whylogs_logger
    w_logger = whylogs_logger()

   


    """ WILDS CAMELYON DATASET """

    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader
    import torchvision.transforms as transforms
    from wilds.datasets.wilds_dataset import WILDSSubset
    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="camelyon17", download=False)

    # Get the training set
    train_data = dataset.get_subset("train")

    img = train_data.dataset[1][0]
    print(type(img))
    from progressbar import progressbar
    #  self.dataset[self.indices[idx]]
    # pil_images = [ train_data.dataset[idx][0] for idx in progressbar(train_data.indices)]
    

    TMP_PATH ='/home/jinglewsl/evoila/sandbox/whylogs_v1/ml-image-drift-monitoring/src/modules/whylogs/data'
    # train_camelyon_profile= w_logger.log_pil_data(
    #     data_directory_path=TMP_PATH,
    #     pil_data_arr=pil_images,
    #     batch_size=len(pil_images))
    
    # w_logger.serialize_profile(profile=train_camelyon_profile,binary_name='train_camelyon_profile',data_directory_path=TMP_PATH)
    
    test_data = dataset.get_subset("test")

    pil_images_train =  [ train_data.dataset[idx][0] for idx in progressbar(train_data.indices)]

    test_camelyon_profile = w_logger.log_pil_data(
        data_directory_path=TMP_PATH,
        pil_data_arr=pil_images_train,
        batch_size=len(pil_images_train)
    )
    
    

    """ my testing """
    # print('len: {}'.format(train_data.__len__()))
    # print(img) 
    # outfile = "/home/jinglewsl/evoila/sandbox/whylogs_v1/ml-image-drift-monitoring/output.png"
    # try:
    #     img.save(outfile)
    # except Exception as e:
    #     print(e)
    # Prepare the standard data loader
    # train_loader = get_train_loader("standard", train_data, batch_size=16)

    # print(type(train_loader))
    from torch.utils.data.dataloader import DataLoader

    """ TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>"""
    # for labeled_batch in train_loader:
    #     print(labeled_batch)
        # x,y, metadata = labeled_batch
        # print('x:{} y:{} metadata:{}'.format(x,y,metadata))

    
    # # (Optional) Load unlabeled data
    # dataset = get_dataset(dataset="iwildcam", download=True, unlabeled=True)
    # unlabeled_data = dataset.get_subset(
    #     "test_unlabeled",
    #     transform=transforms.Compose(
    #         [transforms.Resize((448, 448)), transforms.ToTensor()]
    #     ),
    # )
    # unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)

    # # Train loop
    # for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_loader):
    #     x, y, metadata = labeled_batch
    #     unlabeled_x, unlabeled_metadata = unlabeled_batch
        



# ======================================================================================
# call
if __name__ == "__main__":
    main()