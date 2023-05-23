from torch.utils.data import Dataset, DataLoader
from PIL import Image
import skimage
from skimage.color import gray2rgb
from torchvision import transforms,io
import torchvision
import torch
class NoneTransform:
    def __call__(self, im):
        return im
    
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size=output_size

    def __call__(self,image):
        height,width=image.shape[1],image.shape[2]
        if isinstance(self.output_size,int):
            if height>width:
                new_height,new_width=self.output_size*height/width,self.output_size
            else:
                new_height,new_width=self.output_size,self.output_size*width/height

            new_height,new_width=int(new_height),int(new_width)
            image=skimage.transform.resize(image,(new_height,new_width))
        return image



#use torchvision.io class ,all images are read as tensors and rgb values
#returns tensor of size (3, height, width)
#returns label of 0,1
class HumerusDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.labels = self.df['Label'].map({'positive': 1, 'negative': 0})
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path=self.df['path'].iloc[idx]
        # this does not works???
        #open and transform to rgb
        img = io.read_image(path,mode=io.ImageReadMode.RGB)
        # if self.transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Lambda(lambda x:x.repeat(3,1,1)) if img.mode!='RGB' else NoneTransform(),
        #     ])   
        if self.transform is None: 
            self.transform=NoneTransform()
        img = self.transform(img)
        img=torchvision.transforms.functional.convert_image_dtype(img,torch.float32)
        print("img shape:",img.shape,"img dtype:",img.dtype)
        print("label shape:",self.labels.iloc[idx].shape,"label dtype:",self.labels.iloc[idx].dtype)
        return img, self.labels.iloc[idx]
    

#simpleflatten +2layers NN +sigmoid
class BaselineNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size

        self.flatten=torch.nn.Flatten()
        self.nn1=torch.nn.Linear(self.input_size,self.hidden_size)
        self.nn2=torch.nn.Linear(self.hidden_size,self.output_size)


    def forward(self,x):
        print("pre flatten layer:", x.shape,x.dtype)
        x=self.flatten(x)
        print("post flatten layer:", x.shape, x.dtype)
        print(self.nn1.weight.shape,self.nn1.weight.dtype)
        x=self.nn1(x)
        print("post nn1 layer:", x.shape,x.dtype)
        x=torch.nn.functional.relu(x)
        print("post relu layer:", x.shape,x.dtype)
        x=self.nn2(x)
        print("post nn2 layer:", x.shape,x.dtype)
        output=torch.nn.functional.sigmoid(x)
        print("post sigmoid layer:", output.shape,output.dtype)
        print("output:",output)
        return torch.squeeze(output)
        
    def reset_parameters(self):
        self.nn1.reset_parameters()
        self.nn2.reset_parameters()
