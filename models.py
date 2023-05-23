from torch.utils.data import Dataset, DataLoader
from PIL import Image
import skimage
from skimage.color import gray2rgb
from torchvision import transforms,io
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
        return img, self.labels.iloc[idx]