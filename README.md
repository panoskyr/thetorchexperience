# MURA


# Coding hindrances
1. if using torchvision.io.read_image one should not use ToTensor as a transformation
The pic is already read as a Tensor
2. loading the image with pytorch vision produces a tensor of uint8 but the dtype=float32 for all operations .needed to find a way to convert
3. used Tensor.type_as(outputs) in train function to convert from int64 to float32 to pass into the loss function
4. Wrote custom class to use functional transforms
5. Used a Linear Layer with 1 output on top of the resnet model. Also used BCEWithLogitsLoss as a loss function because the sigmoid is not apllied on the linear level. 





# Sources
1. [Densenet with particular loss function](https://github.com/ishanrai05/MURA-stanford/blob/master/notebook/Mura.ipynb)

2. [How to use transfer learning with alex net](https://github.com/madsendennis/notebooks/blob/master/pytorch/3_PyTorch_Transfer_learning.ipynb)

3. [Transfer Learning from resnet ](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet)

# TODO
## Panagiotis
Ensemble class for two already trained models



## Nikos
Description of Multiclass problem that makis said in class



# Experimental Results

## Baseline NN on humerus only
Training params:
Adam, lr=0.001
epochs: 20
Transforms:
Test loss: 2.10770, Test acc: 0.35%
![alt](experiments\loss_simpleNN_notransforms.png)
