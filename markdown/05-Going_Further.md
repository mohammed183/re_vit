::: {.cell .markdown}
# (Optional) Going Further
At this point we acheived our goals that was introduced in the first section of this material. The following experiments will verify the rest of the claims introduced in the claims section. You can try to validate these claims by following the steps described. However, it will not be as easy as the previous part as you will need to solve some of the challenges that was explained in the previous experiment yourself this time (eg. finding the learning rate, deciding which models to use, etc).
:::

::: {.cell .markdown}
***
:::

::: {.cell .markdown}
## Claim 2: The performance of the Vision Transformer on the classification task after fine tuning improves with the increase of the pretraining dataset size

The authors claim that their **Vision Transformer** models can learn more effectively from larger datasets than conventional **CNNs**, which enhances their performance on downstream tasks. This means that the vision transformer can transfer the knowledge learned from previous datasets to new ones more effectively than the **ResNet** models.

To demonstrate their claim, the authors compared the ResNet models and the vision transformer models that were pre-trained on three different datasets: **ImageNet**, **ImageNet-21k** and **JFT-300M**. They then fine-tuned the models on the ImageNet dataset for classification. The figure below shows how the pretraining dataset size affects the test accuracy of the models.

![](assets/claim2.png)


To evaluate the performance of the vision transformer model, the author fine-tuned it on various datasets and presented the results in the tables below. The pre-trained models that are marked in green are publicly available.

Model pre-trained on the **ImageNet** dataset:

<table style=“white-space: nowrap; width=100%”>
	<tr>
		<th style="text-align: center;">Dataset</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-B/16</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-B/32</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-L/16</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-L/32</th>
		<th style="text-align: center;">ViT-H/14</th>
	</tr>
	<tr>
		<td style="text-align: center;">CIFAR-10</td>
		<td style="text-align: center;">98.13</td>
		<td style="text-align: center;">97.77</td>
		<td style="text-align: center;">97.86</td>
		<td style="text-align: center;">97.94</td>
		<td style="text-align: center;"> - </td>
	</tr>
	<tr>
		<td style="text-align: center;">CIFAR-100</td>
		<td style="text-align: center;">87.13</td>
		<td style="text-align: center;">86.31</td>
		<td style="text-align: center;">86.35</td>
		<td style="text-align: center;">87.07</td>
		<td style="text-align: center;"> - </td>
	</tr>
	<tr>
		<td style="text-align: center;">ImageNet</td>
		<td style="text-align: center;">77.91</td>
		<td style="text-align: center;">73.38</td>
		<td style="text-align: center;">76.53</td>
		<td style="text-align: center;">71.16</td>
		<td style="text-align: center;"> - </td>
	</tr>
	<tr>
		<td style="text-align: center;">ImageNet ReaL</td>
		<td style="text-align: center;">83.57</td>
		<td style="text-align: center;">79.56</td>
		<td style="text-align: center;">82.19</td>
		<td style="text-align: center;">77.83</td>
		<td style="text-align: center;"> - </td>
	</tr>
	<tr>
		<td style="text-align: center;">Oxford Flowers-102</td>
		<td style="text-align: center;">89.49</td>
		<td style="text-align: center;">85.43</td>
		<td style="text-align: center;">89.66</td>
		<td style="text-align: center;">86.36</td>
		<td style="text-align: center;"> - </td>
	</tr>
	<tr>
		<td style="text-align: center;">Oxford-IIIT-Pets</td>
		<td style="text-align: center;">93.81</td>
		<td style="text-align: center;">92.04</td>
		<td style="text-align: center;">93.64</td>
		<td style="text-align: center;">91.35</td>
		<td style="text-align: center;"> - </td>
	</tr>
</table>

Model pre-trained on the **ImageNet-21k** dataset:

<table style=“white-space: nowrap; width=100%”>
	<tr>
		<th style="text-align: center;">Dataset</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-B/16</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-B/32</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-L/16</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-L/32</th>
		<th style="background-color: lightgreen; text-align: center;">ViT-H/14</th>
	</tr>
	<tr>
		<td style="text-align: center;">CIFAR-10</td>
		<td style="text-align: center;">98.95</td>
		<td style="text-align: center;">98.79</td>
		<td style="text-align: center;">99.16</td>
		<td style="text-align: center;">99.13</td>
		<td style="text-align: center;">99.27</td>
	</tr>
	<tr>
		<td style="text-align: center;">CIFAR-100</td>
		<td style="text-align: center;">91.67</td>
		<td style="text-align: center;">91.97</td>
		<td style="text-align: center;">93.44</td>
		<td style="text-align: center;">93.04</td>
		<td style="text-align: center;">93.82</td>
	</tr>
	<tr>
		<td style="text-align: center;">ImageNet</td>
		<td style="text-align: center;">83.97</td>
		<td style="text-align: center;">81.28</td>
		<td style="text-align: center;">85.15</td>
		<td style="text-align: center;">80.99</td>
		<td style="text-align: center;">85.13</td>
	</tr>
	<tr>
		<td style="text-align: center;">ImageNet ReaL</td>
		<td style="text-align: center;">88.35</td>
		<td style="text-align: center;">86.63</td>
		<td style="text-align: center;">88.40</td>
		<td style="text-align: center;">85.63</td>
		<td style="text-align: center;">88.70</td>
	</tr>
	<tr>
		<td style="text-align: center;">Oxford Flowers-102</td>
		<td style="text-align: center;">99.38</td>
		<td style="text-align: center;">99.11</td>
		<td style="text-align: center;">99.61</td>
		<td style="text-align: center;">99.19</td>
		<td style="text-align: center;">99.51</td>
	</tr>
	<tr>
		<td style="text-align: center;">Oxford-IIIT-Pets</td>
		<td style="text-align: center;">94.43</td>
		<td style="text-align: center;">93.02</td>
		<td style="text-align: center;">94.73</td>
		<td style="text-align: center;">93.09</td>
		<td style="text-align: center;">94.82</td>
	</tr>
</table>

Model pre-trained on the **JFT-300M** dataset:

<table style=“white-space: nowrap; width=100%”>
	<tr>
		<th style="text-align: center;">Dataset</th>
		<th style="text-align: center;">ViT-B/16</th>
		<th style="text-align: center;">ViT-B/32</th>
		<th style="text-align: center;">ViT-L/16</th>
		<th style="text-align: center;">ViT-L/32</th>
		<th style="text-align: center;">ViT-H/14</th>
	</tr>
	<tr>
		<td style="text-align: center;">CIFAR-10</td>
		<td style="text-align: center;">99.00</td>
		<td style="text-align: center;">98.61</td>
		<td style="text-align: center;">99.38</td>
		<td style="text-align: center;">99.19</td>
		<td style="text-align: center;">99.50</td>
	</tr>
	<tr>
		<td style="text-align: center;">CIFAR-100</td>
		<td style="text-align: center;">91.87</td>
		<td style="text-align: center;">90.49</td>
		<td style="text-align: center;">94.04</td>
		<td style="text-align: center;">92.52</td>
		<td style="text-align: center;">94.55</td>
	</tr>
	<tr>
		<td style="text-align: center;">ImageNet</td>
		<td style="text-align: center;">84.15</td>
		<td style="text-align: center;">80.73</td>
		<td style="text-align: center;">87.12</td>
		<td style="text-align: center;">84.37</td>
		<td style="text-align: center;">88.04</td>
	</tr>
	<tr>
		<td style="text-align: center;">ImageNet ReaL</td>
		<td style="text-align: center;">88.85</td>
		<td style="text-align: center;">86.27</td>
		<td style="text-align: center;">89.99</td>
		<td style="text-align: center;">88.28</td>
		<td style="text-align: center;">90.33</td>
	</tr>
	<tr>
		<td style="text-align: center;">Oxford Flowers-102</td>
		<td style="text-align: center;">99.56</td>
		<td style="text-align: center;">99.27</td>
		<td style="text-align: center;">99.56</td>
		<td style="text-align: center;">99.45</td>
		<td style="text-align: center;">99.68</td>
	</tr>
	<tr>
		<td style="text-align: center;">Oxford-IIIT-Pets</td>
		<td style="text-align: center;">95.80</td>
		<td style="text-align: center;">93.40</td>
		<td style="text-align: center;">97.11</td>
		<td style="text-align: center;">95.83</td>
		<td style="text-align: center;">97.56</td>
	</tr>
</table>



We want to test the qualitative claim by fine-tuning the pre-trained models to classify images in the ImageNet dataset as the authors did and examine how the size of each pretraining dataset influences the final test accuracy. However, we cannot verify the results of the **JFT-300M** pre-trained models as they are not publicly available.

We can only test the quantitative claims for the models pre-trained on **ImageNet-21k**, since the models and dataset for **JFT-300M** are not accessible to the public. The models pre-trained on **ImageNet-1k** are available, but they use a different optimizer (SAM) than the one described in the paper. This means that we might get similar results, but not exactly the same as the paper.

***
:::

::: {.cell .markdown}
## Experiment 2:

In this experiment we want to verify the claim that states that *"The performance of the Vision Transformer on the classification task after fine tuning improves with the increase of the pretraining dataset size"* by using the available pre-trained model as before. For this claim the authors compare the performance after pretraining on three datasets: **ImageNet-1k**, **ImageNet-21k** and **JFT-300M**. However, we cannot reproduce the results related to the **JFT-300M** dataset as before, so we will only use the model pre-trained on the other two datasets. We mainly compare the results on the ImageNet-1k dataset but we can also extend the experiment and fine tune the model on the other datasets but this can be computationly expensive.

***
:::

::: {.cell .markdown}
The models available for this experiment are:

| Model          | Pretrained ImageNet | pre-trained ImageNet-21k |
| :------------: | :-----------------: | :--------------------: |
| ResNet50x1     | Yes                 | Yes                    |
| ResNet101x1    | Yes                 | Yes                    |
| ResNet152x2    | Yes                 | Yes                    |
| ResNet152x4    | Yes                 | Yes                    |
| ViT-B/32       | Yes (SAM)           | Yes                    |
| ViT-B/16       | Yes (SAM)           | Yes                    |
| ViT-L/32       | Yes (SAM)           | Yes                    |
| ViT-L/16       | Yes (SAM)           | Yes                    |

We already have the results for the **ResNet-152x4** and **ViT-L/16** pre-trained on the **ImageNet-21k**.

Notice that the vision transformers pre-trained on the **ImageNet-1k** use a different optimizer than described in the paper which will prevent us from validating the quantitative results of this claim. Moreover, these models are only compatible with the **JAX** framework, not with **PyTorch**. Therefore, we need to use **JAX** to load and use these models. To test the qualitative claim, we can choose any two models per dataset and compare their performance.

***
:::

:::{.cell .markdown}
In this Experiment we will us the following three notebooks:

- [ResNet notebook](03.1-ResNet.ipynb): Thi is the same notebook from the previous experiment: We can change the model name by modifying the `model_name` variable in the code. The available models are listed in this [link](https://console.cloud.google.com/storage/browser/bit_models;tab=objects?prefix=&forceOnObjectsSortingFiltering=false). The models with 'M' in their names are pre-trained on **ImageNet-21k**, while the ones with 'S' are pre-trained on **ImageNet-1k**. The models with both 'M' and 'ILSVRC2012' are finetuned on **ImageNet-1k**, while the others require finetuning.

- [ViT notebook](03.2-ViT.ipynb): This is the same notebook from the previous experiment. We can change the model name by modifying the `model_name` variable in the code. The available models are listed in this [link](https://huggingface.co/models?sort=trending&search=google%2Fvit-). We need to use the (384x384) models for **ImageNet-1k** and the (224x224) models for the other datasets.

- [ViT-JAX notebook](https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb): This is the official ViT notebook published by google. The notebook allows us to evaluate the performance of different Vision Transformer (ViT) models on various image classification datasets using the JAX framework instead of PyTorch. We can change the model name by modifying the `model_name` variable in the code. The available models are listed in this [link](https://console.cloud.google.com/storage/browser/vit_models/sam?authuser=0&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false).

We will use these notebooks to obtain the results for each model and dataset combination and store them in JSON files. 
**Note: We need to be careful when naming the JSON files to avoid overwriting previous results.**

***
:::

::: {.cell .markdown}
After finishing the previous part, run the following cell to reproduce the table and compare it to the one in the claims section.
:::

::: {.cell .code}
```python
# Load the data from both json files and create a table with results
import pandas as pd
import json

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
display(df.style.set_properties(**{'text-align': 'center', 'border': '2px solid black', 'padding': '10px',\
                                   'font-size': '15px'}))
```
:::

::: {.cell .markdown}
***
:::


::: {.cell .markdown}
## Claim 3: The hybrid Vision Transformer can perform better than both baseline and Vision Transformer after fine tuning it to different classification task

The authors propose a **hybrid vision transformer** that combines a **ResNet** backbone with a **vision transformer**, and claim that this model outperforms both the *pure vision transformer* and the pure *CNN* models. They argue that the ResNet layer provides a better feature representation for the transformer, enabling it to learn more information from the images.

To support their claim, the authors conduct various experiments using different models pre-trained on the **JFT-300M** dataset and then fine-tuned on classification tasks on different datasets. They also vary the number of epochs as a hyperparameter and test the sensitivity of their results to this factor. The tables below show the results they obtained for each type of model.

The **ResNet models** results:

<table style=“white-space: nowrap; width=100%”>
	<tr>
		<th style="text-align: center;">Model</th>
		<th style="text-align: center;">Epochs</th>
		<th style="text-align: center;">ImageNet</th>
		<th style="text-align: center;">ImageNet ReaL</th>
		<th style="text-align: center;">CIFAR-10</th>
		<th style="text-align: center;">CIFAR-100</th>
		<th style="text-align: center;">Pets</th>
		<th style="text-align: center;">Flowers</th>
	</tr>
	<tr>
		<td style="text-align: center;">ResNet50x1</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">77.54</td>
		<td style="text-align: center;">84.56</td>
		<td style="text-align: center;">97.67</td>
		<td style="text-align: center;">86.07</td>
		<td style="text-align: center;">91.11</td>
		<td style="text-align: center;">94.26</td>
	</tr>
	<tr>
		<td style="text-align: center;">ResNet50x2</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">82.12</td>
		<td style="text-align: center;">87.94</td>
		<td style="text-align: center;">98.29</td>
		<td style="text-align: center;">89.20</td>
		<td style="text-align: center;">93.43</td>
		<td style="text-align: center;">97.02</td>
	</tr>
	<tr>
		<td style="text-align: center;">ResNet101x1</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">80.67</td>
		<td style="text-align: center;">87.07</td>
		<td style="text-align: center;">98.48</td>
		<td style="text-align: center;">89.17</td>
		<td style="text-align: center;">94.08</td>
		<td style="text-align: center;">95.95</td>
	</tr>
	<tr>
		<td style="text-align: center;">ResNet152x1</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">81.88</td>
		<td style="text-align: center;">87.96</td>
		<td style="text-align: center;">98.82</td>
		<td style="text-align: center;">90.22</td>
		<td style="text-align: center;">94.17</td>
		<td style="text-align: center;">96.94</td>
	</tr>
	<tr>
		<td style="text-align: center;">ResNet152x2</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">84.97</td>
		<td style="text-align: center;">89.69</td>
		<td style="text-align: center;">99.06</td>
		<td style="text-align: center;">92.05</td>
		<td style="text-align: center;">95.37</td>
		<td style="text-align: center;">98.62</td>
	</tr>
	<tr>
		<td style="text-align: center;">ResNet152x2</td>
		<td style="text-align: center;">14</td>
		<td style="text-align: center;">85.56</td>
		<td style="text-align: center;">89.89</td>
		<td style="text-align: center;">99.24</td>
		<td style="text-align: center;">91.92</td>
		<td style="text-align: center;">95.75</td>
		<td style="text-align: center;">98.75</td>
	</tr>
	<tr>
		<td style="text-align: center;">ResNet200x3</td>
		<td style="text-align: center;">14</td>
		<td style="text-align: center;">87.22</td>
		<td style="text-align: center;">90.15</td>
		<td style="text-align: center;">99.34</td>
		<td style="text-align: center;">93.53</td>
		<td style="text-align: center;">96.32</td>
		<td style="text-align: center;">99.04</td>
	</tr>
</table>

The **vision transformer models** results:

<table style=“white-space: nowrap; width=100%”>
	<tr>
		<th style="text-align: center;">Model</th>
		<th style="text-align: center;">Epochs</th>
		<th style="text-align: center;">ImageNet</th>
		<th style="text-align: center;">ImageNet ReaL</th>
		<th style="text-align: center;">CIFAR-10</th>
		<th style="text-align: center;">CIFAR-100</th>
		<th style="text-align: center;">Pets</th>
		<th style="text-align: center;">Flowers</th>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-B/32</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">80.73</td>
		<td style="text-align: center;">86.27</td>
		<td style="text-align: center;">98.61</td>
		<td style="text-align: center;">90.49</td>
		<td style="text-align: center;">93.40</td>
		<td style="text-align: center;">99.27</td>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-B/16</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">84.15</td>
		<td style="text-align: center;">88.85</td>
		<td style="text-align: center;">99.00</td>
		<td style="text-align: center;">91.87</td>
		<td style="text-align: center;">95.80</td>
		<td style="text-align: center;">99.56</td>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-L/32</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">84.37</td>
		<td style="text-align: center;">88.28</td>
		<td style="text-align: center;">99.19</td>
		<td style="text-align: center;">92.52</td>
		<td style="text-align: center;">95.83</td>
		<td style="text-align: center;">99.45</td>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-L/16</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">86.30</td>
		<td style="text-align: center;">89.43</td>
		<td style="text-align: center;">99.38</td>
		<td style="text-align: center;">93.46</td>
		<td style="text-align: center;">96.81</td>
		<td style="text-align: center;">99.66</td>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-L/16</td>
		<td style="text-align: center;">14</td>
		<td style="text-align: center;">87.12</td>
		<td style="text-align: center;">89.99</td>
		<td style="text-align: center;">99.38</td>
		<td style="text-align: center;">94.04</td>
		<td style="text-align: center;">97.11</td>
		<td style="text-align: center;">99.56</td>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-H/14</td>
		<td style="text-align: center;">14</td>
		<td style="text-align: center;">88.08</td>
		<td style="text-align: center;">90.36</td>
		<td style="text-align: center;">99.50</td>
		<td style="text-align: center;">94.71</td>
		<td style="text-align: center;">97.11</td>
		<td style="text-align: center;">99.71</td>
	</tr>
</table>

The **hybrid models** results:

<table style=“white-space: nowrap; width=100%”>
	<tr>
		<th style="text-align: center;">Model</th>
		<th style="text-align: center;">Epochs</th>
		<th style="text-align: center;">ImageNet</th>
		<th style="text-align: center;">ImageNet ReaL</th>
		<th style="text-align: center;">CIFAR-10</th>
		<th style="text-align: center;">CIFAR-100</th>
		<th style="text-align: center;">Pets</th>
		<th style="text-align: center;">Flowers</th>
	</tr>
	<tr>
		<td style="text-align: center;">R50x1+ViT-B/32</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">87.90</td>
		<td style="text-align: center;">89.15</td>
		<td style="text-align: center;">99.01</td>
		<td style="text-align: center;">92.24</td>
		<td style="text-align: center;">95.75</td>
		<td style="text-align: center;">99.46</td>
	</tr>
	<tr>
		<td style="text-align: center;">R50x1+ViT-B/16</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">85.58</td>
		<td style="text-align: center;">89.65</td>
		<td style="text-align: center;">99.14</td>
		<td style="text-align: center;">92.63</td>
		<td style="text-align: center;">96.65</td>
		<td style="text-align: center;">99.40</td>
	</tr>
	<tr>
		<td style="text-align: center;">R50x1+ViT-L/32</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">85.68</td>
		<td style="text-align: center;">89.04</td>
		<td style="text-align: center;">99.24</td>
		<td style="text-align: center;">92.93</td>
		<td style="text-align: center;">96.97</td>
		<td style="text-align: center;">99.43</td>
	</tr>
	<tr>
		<td style="text-align: center;">R50x1+ViT-L/16</td>
		<td style="text-align: center;">7</td>
		<td style="text-align: center;">86.60</td>
		<td style="text-align: center;">89.72</td>
		<td style="text-align: center;">99.18</td>
		<td style="text-align: center;">93.64</td>
		<td style="text-align: center;">97.03</td>
		<td style="text-align: center;">99.40</td>
	</tr>
	<tr>
		<td style="text-align: center;">R50x1+ViT-L/16</td>
		<td style="text-align: center;">14</td>
		<td style="text-align: center;">87.12</td>
		<td style="text-align: center;">89.76</td>
		<td style="text-align: center;">99.31</td>
		<td style="text-align: center;">93.89</td>
		<td style="text-align: center;">97.36</td>
		<td style="text-align: center;">99.11</td>
	</tr>
</table>

We are unable to verify the quantitative results of the previous claim because we do not have access to the pre-trained models or the **JFT-300M** dataset, which is a private dataset that the authors used for pretraining.

We want to test the claim that the **hybrid vision transformer** models can outperform both the **CNN** and **vision transformer** models on different classification tasks. However, we do not have the models or the data that the authors used for pretraining. To address this issue, we will use the models that the authors pre-trained on the **ImageNet-21k** dataset, which is a public dataset. We will fine-tune these models on various classification tasks, such as CIFAR-10, CIFAR-100, Pets, and Flowers. Then, we will compare the test accuracy of each type of model: pure vision transformer, pure CNN, and hybrid vision transformer.

***
:::



::: {.cell .markdown}
## Experiment 3:

In this experiment we want to verify the claim that states that "The hybrid Vision Transformer can perform better than both baseline and Vision Transformer after fine tuning it to different classification task" by using the available pre-trained model as before. The models used by the authors are not available but there other models available that we can use.

***
:::

::: {.cell .markdown}
The models available for this experiment are all pre-trained on the **ImageNet-21k** datasets:

-  R50x1+ViT-B/16 
-  R50x1+ViT-L/32
-  R50x1+ViT-L/16

We can use any of these model and compare it to the results from the previous experiments to validate the qualitative version of the claim. However, we cannot validata the quantitative results as the models pre-trained on the **JFT-300M** dataset are not available.

***
:::

::: {.cell .markdown}
The notebooks used for this experiment:

- [ViT-JAX notebook](https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb): This is the same notebook fromt the previous experiment. We can change the model name by modifying the `model_name` variable in the code. The available models are listed in this [link](https://console.cloud.google.com/storage/browser/vit_models?authuser=0&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false). The model pre-trained on the **ImageNet-21k** only without fine-tuning are available in the `imagenet21k/` folder, while the fine-tuned models on the **ImageNet-1k** are available in the `imagenet21k+imagenet2012/`folder.

**Note: We need to be careful when naming the JSON files to avoid overwriting previous results.**

***
:::

::: {.cell .markdown}
After finishing the previous part, run the following cell to reproduce the table and compare it to the one in the claims section.
:::

::: {.cell .code}
```python
# Load the data from both json files and create a table with results
import pandas as pd
import json

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
display(df.style.set_properties(**{'text-align': 'center', 'border': '2px solid black', 'padding': '10px',\
                                   'font-size': '15px'}))
```
:::

