::: {.cell .markdown}
## Fine tuning the Vision Transformer

In this notebook we use the pretrained ViT-L/16 model on the ImageNet-21k dataset which contains about 14 million images. The model will be finetuned on different datasets which are used for image classification tasks and then compared the performance to the baseline model.

***
:::

::: {.cell .markdown}
We will use the pretrained weights on hugging face which are the same weights provided by the authors but translated to be used in Pytorch.
The next few cells show the functions used for finetuning the **ViT-L/16** model on different datasets.

***
:::

::: {.cell .code}
```python
# install hugging face transformer <<< Move later to requirements >>>
!pip install transformers
```
::: 

::: {.cell .code}
```python
import os
import json
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTModel
from torch.utils.data import random_split
from torchvision import transforms, datasets, models
```
:::

::: {.cell .code}
```python
# Create data loaders for transformer
def get_vit_loaders(dataset="imagenet", batch_size=64):
    """
    This loads the whole dataset into memory and returns train and test data to
    be used by the Vision Transformer
    @param dataset (string): dataset name to load
    @param batch_size (int): batch size for training and testing

    @returns dict() with train and test data loaders with keys `train_loader`, `test_loader`
    """
    # Normalization using channel means
    normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Creating transform function
    train_transform =transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize_transform])

    # Test transformation function
    test_transform =transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize_transform])

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


        model_y = model.classifier(model(data_x).pooler_output)
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
def train_vit_model(title="", loaders, model_name='google/vit-base-patch16-224-in21k',
                         lr=0.003, epochs=10, random_seed=42, save=False):

    # Create experiment directory 
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

    # Load the pre-trained model
    model = ViTModel.from_pretrained(model_name)
    # Create a new linear layer with num_classes
    new_classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    # Assign it to the model.classifier attribute
    model.classifier = new_classifier
    # Move the model to the device
    model = model.to(device)

    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Arrays to hold accuracies
    test_acc = [0]
    train_acc = [0]

    # Iterate over the number of epochs
    for epoch in range(1, epochs + 1):
        model.train()
        print(f"Epoch {epoch}")
        accuracies = []
        losses = []

        # Calculate loss and gradients for models on every training batch
        for batch_idx, (data_x, data_y) in enumerate(loaders["train_loader"]):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()
            model_y = model.classifier(model(data_x).pooler_output)
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


            model_y = model.classifier(model(data_x).pooler_output)
            loss = criterion(model_y, data_y)
            batch_accuracy = get_accuracy(model_y, data_y)

            accuracies.append(batch_accuracy.item())
            losses.append(loss.item())

        # Store test accuracy for plotting
        test_loss = np.mean(losses)
        test_accuracy = np.mean(accuracies)
        test_acc.append(test_accuracy*100)
        print("Test accuracy: {} Test loss: {}".format(test_accuracy, test_loss))

    # Save the final model
    if save:
        torch.save({
            'model': model.state_dict()
        }, os.path.join(experiment_dir, f'Vit-L/16{title}.pt'))

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

::: {.cell .markdown}
### ImageNet

The ImageNet dataset consists of **1000** object classes and contains **1,281,167** training images, **50,000** validation images and **100,000** test images. The images vary in resolution but it is common practice to train deep learning models on sub-sampled images of **256x256**pixels. This dataset is widely used for image classification and localization tasks and has been the benchmark for many state-of-the-art algorithms. 

***
:::

::: {.cell .code}
```python
# Plot some images from the ImageNet dataset
loader = get_vit_loaders(dataset="imagenet", batch_size=32)
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
num_classes = len(loader["train_loader"].dataset.classes)

# Get the fine tuned model on the ImageNet dataset
model = ViTModel.from_pretrained('google/vit-large-patch16-224')
# Move the model to the device
model = model.to(device)
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
with open("experiments/exp1/vit.json", "w") as f:
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
loader = get_vit_loaders(dataset="cifar10", batch_size=32)
plot_images_from_dataloader(loader["train_loader"])
```
:::

::: {.cell .code}
```python
# Fine tune the model on imagenet
train_acc_cifar10, test_acc_cifar10 = train_vit_model(loaders=loader)
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
with open("experiments/exp1/vit.json", "w") as f:
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
loader = get_vit_loaders(dataset="cifar100", batch_size=32)
plot_images_from_dataloader(loader["train_loader"])
```
:::

::: {.cell .code}
```python
# Fine tune the model on imagenet
train_acc_cifar100, test_acc_cifar100 = train_vit_model(loaders=loader)
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
with open("experiments/exp1/vit.json", "w") as f:
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
loader = get_vit_loaders(dataset="oxford_pets", batch_size=32)
plot_images_from_dataloader(loader["train_loader"])
```
:::

::: {.cell .code}
```python
# Fine tune the model on imagenet
train_acc_oxford_pets, test_acc_oxford_pets = train_vit_model(loaders=loader)
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
with open("experiments/exp1/vit.json", "w") as f:
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
loader = get_vit_loaders(dataset="oxford_flowers", batch_size=32)

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
train_acc_oxford_flowers, test_acc_oxford_flowers = train_vit_model(loaders=loader)
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
with open("experiments/exp1/vit.json", "w") as f:
    json.dump(runs, f)
```
:::