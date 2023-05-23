# MURA


# Coding hindrances
1. if using torchvision.io.read_image one should not use ToTensor as a transformation
The pic is already read as a Tensor
2. loading the image with pytorch vision produces a tensor of uint8 but the dtype=float32 for all operations .needed to find a way to convert
3. used Tensor.type_as(outputs) in train function to convert from int64 to float32 to pass into the loss function