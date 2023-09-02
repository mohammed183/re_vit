::: {.cell .markdown}
# Experiments

In this section, we will attempt to verify the qualitative and quantitative aspects of each claim. We will indicate which claims cannot be verified due to the lack of the material published by the authors. We will mainly use pretrained models published to verify these claims. Trying to reproduce such results without pretrained models can be very expensive as the computational requirement is huge.

***
:::

::: {.cell .markdown}
Before we dive into the experiments, let's briefly discuss some of the common challenges that we will encounter in each of them. We will elaborate on these challenges later in the corresponding notebooks, but for now, here are some of the main issues that you should be aware of:

- The models are **very large and memory-intensive**, requiring a GPU with at least 16GB of RAM for the ViT models and at least 24GB of RAM for the ResNet models.
- The authors used a **very large batch size (512) and a huge number of steps**, which can be very costly to reproduce.
- The resolutions reported in the paper are **not consistent with the ones provided in the code**.
- The final fine-tuning learning rate for each dataset is **not reported in the paper**, but only the values used for grid search are given.
- The learning rate scheduler used in the authors' code is **not exactly as described in the paper**.

The following is a subset of table 4 from the paper which includes some of the values mentioned above:

| Dataset            | Steps  | Base LR                    |
| :----------------: | :----: | :------------------------: |
| ImageNet           | 20 000 | {0.003, 0.01, 0.03, 0.06}  |
| CIFAR100           | 10 000 | {0.001, 0.003, 0.01, 0.03} |
| CIFAR10            | 10 000 | {0.001, 0.003, 0.01, 0.03} |
| Oxford-IIIT Pets   | 500    | {0.001, 0.003, 0.01, 0.03} |
| Oxford Flowers-102 | 500    | {0.001, 0.003, 0.01, 0.03} |


***
:::

::: {.cell .markdown}
Before starting any of the experiments, we need to download the **ImageNet-1k** validation data to be able to verify the results on the **ImageNet-1k** dataset as it is *not available* in `torchvision.datasets`.

To access the dataset, you will need a **Hugging Face** account with an access token. You can obtain an access token by following this [tutorial](https://huggingface.co/docs/hub/security-tokens). Once you have an access token, visit the [dataset page](https://huggingface.co/datasets/imagenet-1k), read and accept the terms and conditions, and then run the following cells. Please note that the process may take 10 or more minutes to complete, depending on your internet connection.

**🛑 To avoid getting errors, wait for each cell to finish before running the next cell**
:::

::: {.cell .code}
```python
# Login to hugging face using the token you created
from huggingface_hub import login
login()

"""
If you have Troubles running this cell, you can run this in the terminal

$ pip install huggingface_hub[cli]

$ huggingface-cli login

It will ask for your token, enter it and press enter.
Once you get login successful you can move on to the next cell.
"""
```
:::

::: {.cell .code}
```python
# Download the dataset from hugging face
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="imagenet-1k", filename="data/val_images.tar.gz", repo_type="dataset")
```
:::

::: {.cell .markdown}
The output of the previous cell contains a path. Please copy this path and paste it into the `path` variable in the next cell which will prepare the validation data to be used.
:::

::: {.cell .code}
```python
# Path to dataset file
path = ''

# Move the path to current directory
!cp "$path" val_images.tar.gz
```
:::

::: {.cell .code}
```python
# Create the data/imagenet/val directory and extract the contents of val_images.tar.gz into it
!mkdir -p data/imagenet/val && tar -xzf val_images.tar.gz -C data/imagenet/val
```
:::

::: {.cell .code}
```python
# Download and run the valprep.sh script from the mohammed183/re_vit repository
!cd data/imagenet/val && wget -qO- https://raw.githubusercontent.com/mohammed183/re_vit/main/imagenet_prep.sh | bash
```
:::

::: {.cell .code}
```python
# Remove val_images.tar.gz, you can remove from path using rm "$path"
!rm val_images.tar.gz
```
:::

::: {.cell .markdown}

***
:::

::: {.cell .markdown}
## Primary Experiment:

In this experiment we want to reproduce the claim: *"Vision Transformer outperforms state of the art CNNs on various classification tasks after pretraining on large datasets"* by using the only available pretrained model in the table in that claim and compare it to the other model that are pretrained on the **ImageNet-21k** unlike in the original paper where the other models were pretrained on the **JFT-300M** private dataset.

***
:::

::: {.cell .markdown}
This experiment is divided into two notebooks, which you can use to evaluate the model’s performance on a specific dataset by running the corresponding sections in both notebooks. After running these notebooks, you can create the table from the primary claim using the cell below:

- [ResNet notebook](03.1-ResNet.ipynb): This notebook allows us to evaluate the performance of different **ResNet** models on various image classification datasets. The `model name` can be changed to try different models. The models in this notebook used for this experiment are pretrained on the **ImageNet-21k** dataset and are ready for fine-tuning.

- [ViT notebook](03.2-ViT.ipynb): This notebook allows us to evaluate the performance of different **Vision Transformer (ViT)** models on various image classification datasets. The `model name` can be changed to try different models. The models in this notebook are pretrained on the **ImageNet-21k** dataset and are ready for fine-tuning.

**🛑 If running crashes while GPU is used you will to restart runtime to kill process and empty GPU, if you can't find restart the runtime then use the `nvidia-smi` command in the terminal and kill the process**

***
:::

::: {.cell .markdown}
After running both notebooks, now we can reproduce the table using the results stored in `resnet.json` and `vit.json`
:::

::: {.cell .code}
```python
# Load the data from both json files and create a table with results
import pandas as pd
import json

# Read from json file
with open("experiments/resnet.json", "r") as f:
    resnet = json.load(f)
# Read from json file
with open("experiments/vit.json", "r") as f:
    vit = json.load(f)

overall={}

# Merge resnet and vit dictionaries into one overall dictionary
overall['ViT-L/16'] = vit
overall['ResNet152x4'] = resnet

# Calculate the difference between the results of the two models in the dictionary
overall['Differnce'] = {}
for key in resnet.keys():
    overall['Differnce'][key] = overall['ViT-L/16'][key] - overall['ResNet152x4'][key]

# Create a dataframe with the result to be in a table form
df = pd.DataFrame.from_dict(overall).T

# Display the dataframe
display(df.style.set_properties(**{'text-align': 'center', 'border': '2px solid black', 'padding': '10px',\
                                   'font-size': '15px'}))
```
:::

::: {.cell .markdown}

Create a similar table like the one generated by the previous cell for the models pre-trained on the **JFT-300M** dataset using the results from the paper and compare it to our results. Make sure to include a difference row between the two models.

**Do you think we were able to verify the qualitative version of the claim? 🤔**

***
:::

::: {.cell .markdown}
After running both notebooks, now we can reproduce the table using the results stored in `resnet.json` and `vit.json`
:::

::: {.cell .code}
```python
# Load the data from both json files and create a table with results
import pandas as pd
import json

# Read from json file
with open("experiments/resnet_time.json", "r") as f:
    resnet = json.load(f)
# Read from json file
with open("experiments/vit_time.json", "r") as f:
    vit = json.load(f)

overall={}

# Merge resnet and vit dictionaries into one overall dictionary
overall['ViT-L/16'] = vit
overall['ResNet152x4'] = resnet

# Create a dataframe with the result to be in a table form
df = pd.DataFrame.from_dict(overall).T

# Display the dataframe
display(df.style.set_properties(**{'text-align': 'center', 'border': '2px solid black', 'padding': '10px',\
                                   'font-size': '14px'}))
```
:::

::: {.cell .markdown}

**Can you use the information from the table at the beginning of this notebook and the results from the generated table to calculate the time required to perform a grid search on the learning rates, as described in the paper? 🤓**

***
:::

::: {.cell .markdown}
### Things to try: 🧪

We have experimented with some fine-tuning hyperparameters that yielded good results, but there is still room for improvement in the performance of the models. For example, we can explore:

- Using **different learning rates** to determine the sensitivity of the models to this hyperparameter. A learning rate that is too high or too low can affect the convergence and accuracy of the models.

- Trying different **image resolutions** to see if this improves the results.

- Changing the **number of epochs** to determine its impact on the final results. Altering the number of epochs may lead to better results, but it also increases the risk of overfitting or underfitting.

- Checking the **sensitivity of the model to the random seed** by changing it. The random seed can influence the initialization of the weights and the shuffling of the data, and different seeds may result in different outcomes for the same model and dataset.

**🛑 Please note that trying these options will take some time. You may want to try them only for the Oxford datasets, which require the least amount of time.**

***
:::
