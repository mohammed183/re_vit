::: {.cell .markdown}
# Experiments

In this section, we will attempt to verify the qualitative and quantitative aspects of each claim. We will indicate which claims cannot be verified due to the lack of the material published by the authors. We will mainly use pretrained models published to verify these claims. Trying to reproduce such results without pretrained models can be very expensive as the computational requirement is huge.

***
:::

::: {.cell .markdown}
## Experiment 1:

In this experiment we want to reproduce the claim: *"Vision Transformer outperforms state of the art CNNs on various classification tasks after pretraining on large datasets"* by using the only available pretrained model in the table in that claim and compare it to the other model that are pretrained on the **ImageNet-21k** unlike in the original paper where the other models were trained on the **JFT-300M** private dataset.

***
:::

::: {.cell .markdown}
This Experiment is split into two notebooks:

- [ResNet notebook](ResNet.ipynb): This notebook allows us to evaluate the performance of different ResNet models on various image classification datasets. The model names can be changed to try different models. The models in this notebook are pretrained on the ImageNet-21k dataset and are ready for fine-tuning.

- [ViT notebook](ViT.ipynb): This notebook allows us to evaluate the performance of different Vision Transformer (ViT) models on various image classification datasets. The model names can be changed to try different models. The models in this notebook are pretrained on the ImageNet-21k dataset and are ready for fine-tuning.

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
overall['ResNet152x4'] = resnet
overall['ViT-L/16'] = vit

# Create a dataframe with the result to be in a table form
df = pd.DataFrame.from_dict(overall).T

# Display the dataframe
display(df.style.set_properties(**{'text-align': 'center', 'border': '1px solid black', 'padding': '5px'}))
```
:::

::: {.cell .markdown}
***
:::

::: {.cell .markdown}
# Optional
At this point we acheived our goals that was introduced in the first section of this material. The following experiments will verify the rest of the claims introduced in the claims section. You can try to validate these claims by following the steps described. However, it will not be as easy as the previous part as you will need to solve some of the challenges that was explained in the previous experiment yourself this time (eg. finding the learning rate, deciding which models to use, etc).
:::

::: {.cell .markdown}
***
:::

::: {.cell .markdown}
## Experiment 2:

In this experiment we want to verify the claim that states that *"The performance of the Vision Transformer on the classification task after fine tuning improves with the increase of the pretraining dataset size"* by using the available pretrained model as before. For this claim the authors compare the performance after pretraining on three datasets: **ImageNet-1k**, **ImageNet-21k** and **JFT-300M**. However, we cannot reproduce the results related to the **JFT-300M** dataset as before, so we will only use the model pretrained on the other two datasets. We mainly compare the results on the ImageNet-1k dataset but we can also extend the experiment and fine tune the model on the other datasets but this can be computationly expensive.

***
:::

::: {.cell .markdown}
The models available for this experiment are:

| Model          | Pretrained ImageNet | pretrained ImageNet-21 |
| :------------: | :-----------------: | :--------------------: |
| ResNet50x1     | Yes                 | Yes                    |
| ResNet101x1    | Yes                 | Yes                    |
| ResNet152x2    | Yes                 | Yes                    |
| ResNet152x4    | Yes                 | Yes                    |
| ViT-B/32       | Yes (SAM)           | Yes                    |
| ViT-B/16       | Yes (SAM)           | Yes                    |
| ViT-L/32       | Yes (SAM)           | Yes                    |
| ViT-L/16       | Yes (SAM)           | Yes                    |

We already have the results for the **ResNet-152x4** and **ViT-L/16** pretrained on the **ImageNet-21k**.

Notice that the vision transformers pretrained on the **ImageNet-1k** use a different optimizer than described in the paper which will prevent us from validating the quantitative results of this claim. Moreover, these models are only compatible with the **JAX** framework, not with **PyTorch**. Therefore, we need to use **JAX** to load and use these models. To test the qualitative claim, we can choose any two models per dataset and compare their performance.
***
:::

:::{.cell .markdown}
In this Experiment we will us the following three notebooks:

- [ResNet notebook](ResNet.ipynb): Thi is the same notebook from the previous experiment: We can change the model name by modifying the `model_name` variable in the code. The available models are listed in this [link](https://console.cloud.google.com/storage/browser/bit_models;tab=objects?prefix=&forceOnObjectsSortingFiltering=false). The models with 'M' in their names are pretrained on **ImageNet-21k**, while the ones with 'S' are pretrained on **ImageNet-1k**. The models with both 'M' and 'ILSVRC2012' are finetuned on **ImageNet-1k**, while the others require finetuning.

- [ViT notebook](ViT.ipynb): This is the same notebook from the previous experiment. We can change the model name by modifying the `model_name` variable in the code. The available models are listed in this [link](https://huggingface.co/models?sort=trending&search=google%2Fvit-). We need to use the (384x384) models for **ImageNet-1k** and the (224x224) models for the other datasets.

- [ViT-JAX notebook](): This notebook allows us to evaluate the performance of different Vision Transformer (ViT) models on various image classification datasets using the JAX framework instead of PyTorch. We can change the model name by modifying the `model_name` variable in the code. The available models are listed in this [link](https://console.cloud.google.com/storage/browser/vit_models/sam?authuser=0&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false).

We will use these notebooks to obtain the results for each model and dataset combination and store them in JSON files. 
**Note: We need to be careful when naming the JSON files to avoid overwriting previous results.**

***
:::

::: {.cell .markdown}
After finishing the previous part, run the following cell to reproduce the table and compare it to the one in the claims section.
:::

::: {.cell .code}
```python
# Array to store name of files used to create table
runs = {}
overall = {}
file_names = [] # Add the names if the new files here

# Loop over files
for name in file_names:
	# Read from json file
	with open(f"experiments/{name}.json", "r") as f:
	    runs = json.load(f)
	# Merge dictionary
	overall[name] = runs

# Create a dataframe with the result to be in a table form
df = pd.DataFrame.from_dict(overall).T

# Display the dataframe
display(df.style.set_properties(**{'text-align': 'center', 'border': '1px solid black', 'padding': '5px'}))
```
:::

::: {.cell .markdown}
***
:::

::: {.cell .markdown}
## Experiment 3:

In this experiment we want to verify the claim that states that "The hybrid Vision Transformer can perform better than both baseline and Vision Transformer after fine tuning it to different classification task" by using the available pretrained model as before. The models used by the authors are not available but there other models available that we can use.

***
:::

::: {.cell .markdown}
The models available for this experiment are:

| Model          | Pretrained ImageNet | pretrained ImageNet-21 |
| :------------: | :-----------------: | :--------------------: |
| R50x1+ViT-B/16 | Yes                 | Yes                    |
| R50x1+ViT-L/32 | No                  | Yes                    |
| R50x1+ViT-L/16 | No                  | Yes                    |

We can use any of these model and compare it to the results from the previous experiments to validate the qualitative version of the claim. However, we cannot validata the quantitative results as the models pretrained on the **JFT-300M** dataset are not available.
***
:::

::: {.cell .markdown}
The notebooks used for this experiment:

- [ViT-JAX notebook](): This is the same notebook fromt the previous experiment. We can change the model name by modifying the `model_name` variable in the code. The available models are listed in this [link](https://console.cloud.google.com/storage/browser/vit_models?authuser=0&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false). The model pretrained on the **ImageNet-21k** only without fine-tuning are available in the `imagenet21k/` folder, while the fine-tuned models on the **ImageNet-1k** are available in the `imagenet21k+imagenet2012/`folder.

**Note: We need to be careful when naming the JSON files to avoid overwriting previous results.**

***
:::

::: {.cell .markdown}
After finishing the previous part, run the following cell to reproduce the table and compare it to the one in the claims section.
:::

::: {.cell .code}
```python
# Array to store name of files used to create table
runs = {}
overall = {}
file_names = [] # Add the names of the new files here

# Loop over files
for name in file_names:
	# Read from json file
	with open(f"experiments/{name}.json", "r") as f:
	    runs = json.load(f)
	# Merge dictionary
	overall[name] = runs

# Create a dataframe with the result to be in a table form
df = pd.DataFrame.from_dict(overall).T

# Display the dataframe
display(df.style.set_properties(**{'text-align': 'center', 'border': '1px solid black', 'padding': '5px'}))
```
:::