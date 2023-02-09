import torch 
import torchdrift 
import torchvision
from torch.utils.data import Dataset
def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(768,1024)),
    # torchvision.transforms.CenterCrop(size=(224, 224)),
    torchvision.transforms.ToTensor()])
val_dataset = torchvision.datasets.ImageFolder('/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/landscape_data_raw/landscape_images/',transform=val_transform)

def collate_fn(self, batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        if self.additional_transform:
            batch = (self.additional_transform(batch[0]), *batch[1:])
        return batch    

                                       
def val_dataloader(dataset: Dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=200,num_workers=1, shuffle=True, collate_fn=None)       


inputs, _ = next(iter(val_dataloader(val_dataset)))



inputs_ood = corruption_function(inputs)

# define a transform to convert a tensor to PIL image
transform = torchvision.transforms.ToPILImage()



# make images of the tensor
imgs =  [transform(tensor) for tensor in inputs]

imgs_blurred = [transform(tensor) for tensor in inputs_ood]

i=0
for img in imgs:
    i +=1
    outfile = "/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/tensor_landscape/"+ str(i) + ".png"
    try:
        img.save(outfile)
    except Exception as e:
        print(e)

i=0
for img in imgs_blurred:
    i +=1
    outfile = "/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/blurred_tensor_landscape/"+ str(i) + ".png"
    try:
        img.save(outfile)
    except Exception as e:
        print(e)