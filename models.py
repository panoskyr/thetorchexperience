from torch.utils.data import Dataset, DataLoader
from PIL import Image
import skimage
from skimage.color import gray2rgb
from torchvision import transforms,io
import torchvision
import torch
from torch.nn import Sigmoid

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

#only available in functional so write class to use in transforms compose

class HistEqualizationTransform:
    def __call__(self, im):
        return torchvision.transforms.functional.equalize(im)
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
        #print("img shape:",img.shape,"img dtype:",img.dtype)
        #print("label shape:",self.labels.iloc[idx].shape,"label dtype:",self.labels.iloc[idx].dtype)
        return img, self.labels.iloc[idx]
    
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
class class6_Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        if self.transform is None:
            self.transform = ResNet18_Weights.DEFAULT.transforms()
        else:
            self.transform = transform
        #mapping alphabetically
        self.labels = self.df['Bone'].map({
            'XR_ELBOW': 0,
            'XR_FINGER': 1,
            'XR_FOREARM': 2,
            'XR_HAND': 3,
            'XR_HUMERUS': 4,
            'XR_SHOULDER': 5,
            'XR_WRIST': 6
        })
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path=self.df['path'].iloc[idx]
        pil_img=Image.open(path).convert('RGB')
        #print("pil_img shape:",pil_img.size,"pil_img mode:",pil_img.mode)
        img=self.transform(pil_img)
        img=torchvision.transforms.functional.convert_image_dtype(img,torch.float32)
        return img, self.labels.iloc[idx]


# a class that will train a neural network with a layer that will distinguish the type of images and 
# with the resnet unfrozen
# then we will fit a new linear head to the resnet and train it on all images with the new head
class ResnetTransfer(torch.nn.Module):
    def __init__(self, train_resnet=True):
        super().__init__()

        self.train_resnet = train_resnet

        # first training pass with resnet unfrozen
        # to take info about classes
        # initialize with pretrained weights
        weights=torchvision.models.ResNet18_Weights.DEFAULT
        resnet18 = torchvision.models.resnet18(weights=weights)
        self.resnet=resnet18
        # allow training of parameters
        for param in self.resnet.parameters():
            param.requires_grad=True

        #add a head to the resnet that outputs 7 class probabilities
        self.resnet.fc=torch.nn.Linear(self.resnet.fc.in_features,7)
        for param in self.resnet.fc.parameters():
            param.requires_grad=True
            
        self.softmax=torch.nn.Softmax(dim=1)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self, x):
        x=self.resnet(x)
        if self.train_resnet is True:
            x=self.softmax(x)
        else:
            x =self.sigmoid(x)
        return x

    def trigger_phase2(self):

        if self.train_resnet is True:
            self.train_resnet = False
            
            #we keep the previous training of resnet
            # and add a new head that will output 1 class probability
            for param in self.resnet.parameters():
                param.requires_grad=False
            
            # the new head will be trained from scratch
            self.resnet.fc=torch.nn.Linear(self.resnet.fc.in_features,1)

        else:
            return

    def __str__(self):
        return f"ResnetTransfer(frozen={self.train_resnet}, output_size={self.resnet.fc.out_features})"




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
        #print("pre flatten layer:", x.shape,x.dtype)
        x=self.flatten(x)
        #print("post flatten layer:", x.shape, x.dtype)
        #print(self.nn1.weight.shape,self.nn1.weight.dtype)
        x=self.nn1(x)
        #print("post nn1 layer:", x.shape,x.dtype)
        x=torch.nn.functional.relu(x)
        #print("post relu layer:", x.shape,x.dtype)
        x=self.nn2(x)
        #print("post nn2 layer:", x.shape,x.dtype)
        output=torch.nn.functional.sigmoid(x)
        #print("post sigmoid layer:", output.shape,output.dtype)
        #print("output:",output)
        return torch.squeeze(output)
        
    def reset_parameters(self):
        self.nn1.reset_parameters()
        self.nn2.reset_parameters()



class myEnsemble(torch.nn.Module):
    def __init__(self, modelA, modelB, nb_classes=2):
        super(myEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = None
        self.modelB.fc = None
        # Create new classifier
        self.classifier = torch.nn.Linear(2048+512, nb_classes)