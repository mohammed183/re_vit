::: {.cell .markdown}
## Fine tuning the ResNet model

In this notebook we use the pretrained ResNet-152x4 model on the ImageNet-21k dataset which contains about 14 million images. The model will be finetuned on different datasets which are used for image classification tasks and then used as baseline model:

***
:::

::: {.cell .markdown}
We will use the pretrained weights from the official github repository.

***
:::

::: {.cell .code}
```python
import os
import json
import torch
import numpy as np
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
```
::: 

::: {.cell .code}
```python
def get_res_loaders(dataset="imagenet", batch_size=4):
    """
    This loads the whole dataset into memory and returns train and test data to
    be used by the ResNet model
    @param dataset (string): dataset name to load
    @param batch_size (int): batch size for training and testing

    @returns dict() with train and test data loaders with keys `train_loader`, `test_loader`
    """
    # Normalization using channel means
    normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Creating transform function
    train_transform =transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), normalize_transform])

    # Test transformation function
    test_transform =transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), normalize_transform])

    # Load the dataset from torchvision datasets 
    if dataset == "imagenet":
        # Load ImageNet
        original_train_dataset = datasets.ImageNet(root=os.path.join('data', 'imagenet_data'),
                                             split='train', transform=train_transform, download=True)
        original_test_dataset = datasets.ImageNet(root=os.path.join('data', 'imagenet_data'),
                                             split='val', transform=test_transform, download=True)
    elif dataset == "cifar10":
        # Load CIFAR-10 
        original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=True, transform=train_transform, download=True)
        original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)
    elif dataset == "cifar100":
        # Load CIFAR-100 
        original_train_dataset = datasets.CIFAR100(root=os.path.join('data', 'cifar100_data'),
                                             train=True, transform=train_transform, download=True)
        original_test_dataset = datasets.CIFAR100(root=os.path.join('data', 'cifar100_data'),
                                             train=False, transform=test_transform, download=True)
    elif dataset == "oxford_pets":
        # Load Oxford-IIIT Pets 
        original_train_dataset = datasets.OxfordPets(root=os.path.join('data', 'oxford_pets_data'),
                                             image_set='train', transform=train_transform, download=True)
        original_test_dataset = datasets.OxfordPets(root=os.path.join('data', 'oxford_pets_data'),
                                             image_set='test', transform=test_transform, download=True)
    elif dataset == "oxford_flowers":
        # Load Oxford Flowers-102
        original_train_dataset = datasets.OxfordFlowers102(root=os.path.join('data', 'oxford_flowers_102_data'),
                                             split='train', transform=train_transform, download=True)
        original_test_dataset = datasets.OxfordFlowers102(root=os.path.join('data', 'oxford_flowers_102_data'),
                                             split='test', transform=test_transform, download=True)
    else:
        # Raise an error if the dataset is not valid
        raise ValueError("Invalid dataset name. Please choose one of the following: imagenet, cifar10, \
         cifar100, oxford_pets, oxford_flowers")

    # Creating data loaders
    loader_args = {
        "batch_size": batch_size,
    }

    train_loader = torch.utils.data.DataLoader(
        dataset=original_train_dataset,
        shuffle=True,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loader": train_loader,
            "test_loader": test_loader}
```
::: 

::: {.cell .code}
```python
class StdConv2d(nn.Conv2d):
    """
    A custom convolutional layer that standardizes the weights before applying them.
    """
    def forward(self, x):
        w = self.weight
        # Compute the variance and mean of the weights along the channel, height and width dimensions
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        # Standardize the weights by subtracting the mean and dividing by the standard deviation
        w = (w - m) / torch.sqrt(v + 1e-10)
        # Apply the standardized weights
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

# helper function to create a 3x3 convolutional layer with standardization.    
def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)

# helper function to create a 1x1 convolutional layer with standardization.
def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)

# helper function to convert the weight format from TensorFlow (HWIO) to PyTorch (OIHW).
def tf2th(conv_weights):
    if conv_weights.ndim == 4:
        # Transpose the dimensions from height-width-input-output to output-input-height-width
        conv_weights = np.transpose(conv_weights, [3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """
    A custom residual block that uses pre-activation and group normalization.
    This is based on the paper "Identity Mappings in Deep Residual Networks" by Kaiming He et al.
    https://arxiv.org/abs/1603.05027
    However, this implementation differs from the original one by putting
    the stride on the 3x3 convolution instead of the 1x1 convolution.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        # Set the output channels to be the same as the input channels if not specified
        cout = cout or cin
        # Set the middle channels to be one fourth of the output channels if not specified
        cmid = cmid or cout//4

        # Define the group normalization and convolution layers for the block
        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original ResNetv2 has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        # Define the projection layer for the residual branch if needed
        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        # Conv'ed branch
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        # If there is a projection layer, apply it to the output of the first activation
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # The first block has already applied pre-act before splitting, see Appendix.
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))
        
        # Add the residual branch to the conv'ed branch and return
        return out + residual
    
    # helper function to load the weights from a TensorFlow model
    def load_from(self, weights, prefix=''):
        with torch.no_grad():
            # Copy the weights for each layer from the TensorFlow model to the PyTorch model
            self.conv1.weight.copy_(tf2th(weights[prefix + 'a/standardized_conv2d/kernel']))
            self.conv2.weight.copy_(tf2th(weights[prefix + 'b/standardized_conv2d/kernel']))
            self.conv3.weight.copy_(tf2th(weights[prefix + 'c/standardized_conv2d/kernel']))
            self.gn1.weight.copy_(tf2th(weights[prefix + 'a/group_norm/gamma']))
            self.gn2.weight.copy_(tf2th(weights[prefix + 'b/group_norm/gamma']))
            self.gn3.weight.copy_(tf2th(weights[prefix + 'c/group_norm/gamma']))
            self.gn1.bias.copy_(tf2th(weights[prefix + 'a/group_norm/beta']))
            self.gn2.bias.copy_(tf2th(weights[prefix + 'b/group_norm/beta']))
            self.gn3.bias.copy_(tf2th(weights[prefix + 'c/group_norm/beta']))
            # If there is a projection layer, copy its weights as well
            if hasattr(self, 'downsample'):
                self.downsample.weight.copy_(tf2th(weights[prefix + 'a/proj/standardized_conv2d/kernel']))
        
        # Return the PyTorch model with loaded weights
        return self

# ResNet Class
class ResNetV2(nn.Module):
    BLOCK_UNITS = {
        'r50': [3, 4, 6, 3],
        'r101': [3, 4, 23, 3],
        'r152': [3, 8, 36, 3],
    }

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('padp', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            # The following is subtly not the same!
            #('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin= 64*wf, cout=256*wf, cmid=64*wf))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin= 512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
            ))),
        ]))

        self.zero_head = zero_head
        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048*wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
            ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
        ]))

    def forward(self, x):
        x = self.head(self.body(self.root(x)))
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[...,0,0]

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))
            self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))
                self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')
        return self
```
::: 

::: {.cell .code}
```python
# Function takes predictions and true values to return accuracies
def get_accuracy(logit, true_y):
    pred_y = torch.argmax(logit, dim=1)
    return (pred_y == true_y).float().mean()

# This Function is used to evaluate the model
def evaluate_on_test(model, test_loader, device):
    # Evaluate the model on all the test batches
    accuracies = []
    losses = []
    model.eval()
    for batch_idx, (data_x, data_y) in enumerate(test_loader):
        data_x = data_x.to(device)
        data_y = data_y.to(device)


        model_y = model(data_x)
        loss = criterion(model_y, data_y)
        batch_accuracy = get_accuracy(model_y, data_y)

        accuracies.append(batch_accuracy.item())
        losses.append(loss.item())

    # Store test accuracy for plotting
    test_loss = np.mean(losses)
    test_accuracy = np.mean(accuracies)
    test_acc.append(test_accuracy*100)
    return test_accuracy, test_loss
```
:::

::: {.cell .code}
```python
# Function to train the model and return train and test accuracies
def train_res_model(title='', loaders, model_name='r152', bit_model="BiT-M-R152x4",
                       width_factor=4, lr=0.001, epochs=10, random_seed=42, save=False):

    # Create experiment directory name if none
    experiment_dir = os.path.join('experiments/exp1', title)

    # make experiment directory
    os.makedirs(experiment_dir, exist_ok=True)

    # Set the seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA Recognized")
    else:
        device = torch.device('cpu')

    # Get num_classes
    num_classes = len(loaders["train_loader"].dataset.classes)

    # Get weights
    print(f"Loading {bit_model} weights...")
    weights = get_weights(bit_model)
    print("Weight successfully loaded")

    # Initialize the ResNet model
    model = ResNetV2(ResNetV2.BLOCK_UNITS[model_name], width_factor=width_factor, head_size=num_classes, zero_head=True)
    model.load_from(weights)
    model.to(device);

    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Define a custom function for the learning rate scheduler
    def custom_scheduler(epoch):
        stairs_params = [3e-3, 200, 3e-4, 300, 3e-5, 400, 3e-6, 500, None]

        # Define the parameters for the rampup function
        rampup_params = [100, lr]

        # Compute the learning rate based on the step number
        if epoch < rampup_params[0]:
            lr = epoch / rampup_params[0] * rampup_params[1]
        else:
            lr = rampup_params[1]
            for i in range(0, len(stairs_params), 2):
                if epoch < stairs_params[i]:
                    break
                lr = stairs_params[i+1]

        return lr / rampup_params[1]
    # Create the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Arrays to hold accuracies
    test_acc = []
    train_acc = []

    # Iterate over the number of epochs
    for epoch in range(1, epochs + 1):
        # Make model params trainable
        model.train()
        print(f"Epoch {epoch}")
        accuracies = []
        losses = []

        # Calculate loss and gradients for models on every training batch
        for batch_idx, (data_x, data_y) in enumerate(loaders["train_loader"]):
            # Update the learning rate
            scheduler.step()
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()
            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)

            # Perform back propagation
            loss.backward()
            optimizer.step()

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())

        # Store training accuracy for plotting
        train_loss = np.mean(losses)
        train_accuracy = np.mean(accuracies)
        train_acc.append(train_accuracy*100)

        print("Train accuracy: {} Train loss: {}".format(train_accuracy, train_loss))

        # Evaluate the model on all the test batches
        accuracies = []
        losses = []
        model.eval()
        for batch_idx, (data_x, data_y) in enumerate(loaders["test_loader"]):
            data_x = data_x.to(device)
            data_y = data_y.to(device)


            model_y = model(data_x)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())

        # Store test accuracy for plotting
        test_loss = np.mean(losses)
        test_accuracy = np.mean(accuracies)
        test_acc.append(test_accuracy*100)
        print("Test accuracy: {} Test loss: {}".format(test_accuracy, test_loss))

    if save:
        # Save the final model
        torch.save({
            'model': model.state_dict()
        }, os.path.join(experiment_dir, f'{bit_model}.pt'))

    # return the accuracies
    return train_acc, test_acc
```
::: 

::: {.cell .code}
```python
# Define a function that takes a dataloader as parameter and plots 2 rows each contains 5 images
def plot_images_from_dataloader(dataloader):
    # Get the first batch of images and labels from the dataloader
    images, labels = next(iter(dataloader))
    classes = dataloader.dataset.classes
    # Create a figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    # Loop over the axes and plot each image
    for i, ax in enumerate(axes.flat):
        # Get the image and label at index i
        image = images[i]
        label = classes[labels[i]]
        # Unnormalize the image
        image = image / 2 + 0.5
        # Convert the image to numpy array
        image = image.numpy()
        # Transpose the image to have the channel dimension last
        image = np.transpose(image, (1, 2, 0))
        # Plot the image on the axis
        ax.imshow(image)
        # Set the title of the axis to the label
        ax.set_title(label)
        # Turn off the axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    # Show the plot
    plt.show()
```
:::

::: {.cell .code}
```python
# Create a directory to hold the pretrained models
!mkdir -p pretrained_models

# Download the ResNet152x4 model
![ -e pretrained_models/BiT-M-R152x4.npz ] || \
curl -L -o pretrained_models/BiT-M-R152x4.npz "https://storage.googleapis.com/bit_models/BiT-M-R152x4.npz"

# Download the ResNet152x4 model fine tuned on ImageNet
![ -e pretrained_models/BiT-M-R152x4-ILSVRC2012.npz ] || \
curl -L -o pretrained_models/BiT-M-R152x4-ILSVRC2012.npz "https://storage.googleapis.com/bit_models/BiT-M-R152x4-ILSVRC2012.npz"
```
:::

::: {.cell .code}
```python
def get_weights(bit_variant):
    return np.load(f'pretrained_models/{bit_variant}.npz')
```
:::

::: {.cell .markdown}
### ImageNet

The ImageNet dataset consists of **1000** object classes and contains **1,281,167** training images, **50,000** validation images and **100,000** test images. The images vary in resolution but it is common practice to train deep learning models on sub-sampled images of **256x256**pixels. This dataset is widely used for image classification and localization tasks and has been the benchmark for many state-of-the-art algorithms. 

***
:::

::: {.cell .code}
```python
# Plot some images from the ImageNet dataset
loader = get_res_loaders(dataset="imagenet", batch_size=32)
plot_images_from_dataloader(loader["test_loader"])
```
:::

::: {.cell .code}
```python
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("CUDA Recognized")
else:
    device = torch.device('cpu')

# Get num_classes
num_classes = len(loaders["train_loader"].dataset.classes)

# Get weights
print(f"Loading BiT-M-R152x4-ILSVRC2012 weights...")
weights = get_weights("BiT-M-R152x4-ILSVRC2012")
print("Weight successfully loaded")

# Initialize the ResNet model
model = ResNetV2(ResNetV2.BLOCK_UNITS['r152'], width_factor=4, head_size=num_classes, zero_head=True)
model.load_from(weights)
model.to(device);
```
:::

::: {.cell .code}
```python
# Print the Performance of the Ready fine tuned model
train_acc_imagenet, _ = evaluate_on_test(model, loader["train_loader"], device)
test_acc_imagenet, _ = evaluate_on_test(model, loader["test_loader"], device)
```
:::

::: {.cell .code}
```python
# Add the results to a dictionary
runs["imagenet"] = { 'training_accuracy' : train_acc_imagenet,
                       'test_accuracy' : test_acc_imagenet,
                     }
```
:::

::: {.cell .code}
```python
# Save the outputs in a json file
with open("experiments/exp1/res.json", "w") as f:
    json.dump(runs, f)
```
:::

::: {.cell .markdown}
### CIFAR-10

The CIFAR-10 dataset consists of **60,000 32x32** color images in **10** different classes. The 10 classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. There are **6,000** images per class, with **5,000** for training and **1,000** for testing. It is a popular benchmark for image classification and deep learning research. 

***
:::

::: {.cell .code}
```python
# Plot some images from the CIFAR-10 dataset
loader = get_res_loaders(dataset="cifar10", batch_size=32)
plot_images_from_dataloader(loader["train_loader"])
```
:::

::: {.cell .code}
```python
# Fine tune the model on imagenet
train_acc_cifar10, test_acc_cifar10 = train_res_model(loaders=loader)
```
:::

::: {.cell .code}
```python
# Add the results to a dictionary
runs["cifar10"] = { 'training_accuracy' : train_acc_cifar10,
                       'test_accuracy' : test_acc_cifar10,
                     }
```
:::

::: {.cell .code}
```python
# Save the outputs in a json file
with open("experiments/exp1/res.json", "w") as f:
    json.dump(runs, f)
```
:::

::: {.cell .markdown}
### CIFAR-100

The CIFAR-100 dataset consists of **60,000 32x32** color images in **100** different classes. The 100 classes are grouped into 20 superclasses, such as aquatic mammals, flowers, insects, vehicles, etc. There are **600** images per class, with **500** for training and **100** for testing. It is also a commonly benchmark for image classification and deep learning research.

***
:::

::: {.cell .code}
```python
# Plot some images from the CIFAR-100 dataset
loader = get_res_loaders(dataset="cifar100", batch_size=32)
plot_images_from_dataloader(loader["train_loader"])
```
:::

::: {.cell .code}
```python
# Fine tune the model on imagenet
train_acc_cifar100, test_acc_cifar100 = train_res_model(loaders=loader)
```
:::

::: {.cell .code}
```python
# Add the results to a dictionary
runs["cifar100"] = { 'training_accuracy' : train_acc_cifar100,
                       'test_accuracy' : test_acc_cifar100,
                     }
```
:::

::: {.cell .code}
```python
# Save the outputs in a json file
with open("experiments/exp1/res.json", "w") as f:
    json.dump(runs, f)
```
:::

::: {.cell .markdown}
### Oxford-IIIT Pets

The Oxford-IIIT Pets is a **37** category pet dataset with roughly **200** images for each class created by the Visual Geometry Group at Oxford. The images have large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI (region of interest), and pixel level trimap segmentation. The dataset is useful for fine-grained image classification and segmentation tasks.

***
:::

::: {.cell .code}
```python
# Plot some images from the Oxford-IIIT Pets dataset
loader = get_res_loaders(dataset="oxford_pets", batch_size=32)
plot_images_from_dataloader(loader["train_loader"])
```
:::

::: {.cell .code}
```python
# Fine tune the model on imagenet
train_acc_oxford_pets, test_acc_oxford_pets = train_res_model(loaders=loader)
```
:::

::: {.cell .code}
```python
# Add the results to a dictionary
runs["oxford_pets"] = { 'training_accuracy' : train_acc_oxford_pets,
                       'test_accuracy' : test_acc_oxford_pets,
                     }
```
:::

::: {.cell .code}
```python
# Save the outputs in a json file
with open("experiments/exp1/res.json", "w") as f:
    json.dump(runs, f)
```
:::

::: {.cell .markdown}
### Oxford Flowers-102

The Oxford Flowers-102 dataset consists of **102** flower categories commonly occurring in the United Kingdom. Each class consists of between **40 and 258** images. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset also provides image labels, segmentations, and distances based on shape and color features.

***
:::

::: {.cell .code}
```python
# Plot some images from the Oxford Flowers-102 Pets dataset
loader = get_res_loaders(dataset="oxford_flowers", batch_size=32)

# We initialize the flowers names as they are not on Pytorch (used for plotting)
flower_classes = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle',
 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower',
 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary',
 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke',
 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy',
 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura',
 'geranium', 'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy',
 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily',
 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia',
 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea',
 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']

# Save the Class names in the dataset
loader["train_loader"].dataset.classes = flower_classes
# Plot dataset
plot_images_from_dataloader(loader["train_loader"])
```
:::

::: {.cell .code}
```python
# Fine tune the model on imagenet
train_acc_oxford_flowers, test_acc_oxford_flowers = train_res_model(loaders=loader)
```
:::

::: {.cell .code}
```python
# Add the results to a dictionary
runs["oxford_flowers"] = { 'training_accuracy' : train_acc_oxford_flowers,
                       'test_accuracy' : test_acc_oxford_flowers,
                     }
```
:::

::: {.cell .code}
```python
# Save the outputs in a json file
with open("experiments/exp1/res.json", "w") as f:
    json.dump(runs, f)
```
:::