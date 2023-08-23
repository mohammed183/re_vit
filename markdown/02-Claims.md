::: {.cell .markdown}
# Primary Claims

The paper assesses the performance of the **ResNet**, **ViT** and **hybrid** models on various image classification tasks, and makes some claims based on the assessment. In this section, we present these claims and propose a way to test them using the **pre-trained** models that the authors have provided.The table below shows the different variants of the models that the authors use. The authors also use a notation to refer to the models, such as *ViT-L/16*, which means the “Large” variant with an input patch size of 16 × 16. They also use an improved ResNet as the baseline model and refer to it as [*"ResNet (BiT)"*](https://arxiv.org/abs/1912.11370) which has published models on this [GitHub](https://github.com/google-research/big_transfer).

| Model     | Layers | Hidden size D | MLP size | Heads | Params |
| :-------: | :----: | :-----------: | :------: | :---: | :----: |
| ViT-Base  | 12     | 768           | 3072     | 12    | 86M    |
| ViT-Large | 24     | 1024          | 4096     | 16    | 307M   |
| ViT-Huge  | 32     | 1280          | 5120     | 16    | 632M   |


We want to **evaluate** the **claims** made by the **vision transformer paper**, both **qualitatively and quantitatively**. However, we face some **difficulties** in doing so. First, some of the **pretrained models** that the authors used are **not publicly available**. Second, we do not have enough **computational resources** to train the models from scratch. Third, some of the **datasets** that the authors used for pretraining are **private and inaccessible**. Therefore, we cannot **reproduce all the results** of the paper.

**How do you think we can deal with the previous problems?** 🤔

***
:::

::: {.cell .markdown}
## Claim 1: Vision Transformer outperforms state of the art CNNs on various classification tasks after pretraining on large datasets

This claim suggests that the vision transformer can leverage more knowledge from large datasets than CNN models during pretraining and transfer this knowledge to the fine tuning task. This implies that the vision transformer can achieve higher or comparable accuracies to the state of the art models on different classification tasks.

The authors support their claim by pretraining three versions of the vision transformer and evaluating their performance on various benchmarks, as shown in the table below. However, they use the **JFT-300M** dataset for pretraining, which is a private dataset owned by Google and *not available* to the public. Also, the authors do not share the models pretrained on the **JFT-300M** dataset. *(BiT-L is a ResNet152x4 model)*

<table style="width: 100%;">
	<tr>
		<th style="text-align: center;">Model</th>
		<th style="text-align: center;">ImageNet</th>
		<th style="text-align: center;">ImageNet ReaL</th>
		<th style="text-align: center;">CIFAR-10</th>
		<th style="text-align: center;">CIFAR-100</th>
		<th style="text-align: center;">Oxford-IIIT Pets</th>
		<th style="text-align: center;">Oxford Flowers-102</th>
		<th style="text-align: center;">VTAB (19 tasks)</th>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-H/14 (JFT)</td>
		<td style="text-align: center;">88.55</td>
		<td style="text-align: center;">90.72</td>
		<td style="text-align: center;">99.50</td>
		<td style="text-align: center;">94.55</td>
		<td style="text-align: center;">97.56</td>
		<td style="text-align: center;">99.68</td>
		<td style="text-align: center;">77.63</td>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-L/16 (JFT)</td>
		<td style="text-align: center;">87.76</td>
		<td style="text-align: center;">90.54</td>
		<td style="text-align: center;">99.42</td>
		<td style="text-align: center;">93.90</td>
		<td style="text-align: center;">97.32</td>
		<td style="text-align: center;">99.74</td>
		<td style="text-align: center;">76.28</td>
	</tr>
	<tr style="background-color: lightgreen;">
		<td style="text-align: center;">ViT-L/16 (I21k)</td>
		<td style="text-align: center;">85.30</td>
		<td style="text-align: center;">88.62</td>
		<td style="text-align: center;">99.15</td>
		<td style="text-align: center;">93.25</td>
		<td style="text-align: center;">94.67</td>
		<td style="text-align: center;">99.61</td>
		<td style="text-align: center;">72.72</td>
	</tr>
	<tr>
		<td style="text-align: center;">BiT-L (JFT)</td>
		<td style="text-align: center;">87.54</td>
		<td style="text-align: center;">90.54</td>
		<td style="text-align: center;">99.37</td>
		<td style="text-align: center;">93.51</td>
		<td style="text-align: center;">96.62</td>
		<td style="text-align: center;">99.63</td>
		<td style="text-align: center;">76.29</td>
	</tr>
	<tr style=“white-space: nowrap;”>
		<td style="text-align: center;">Noisy Student</td>
		<td style="text-align: center;">88.4</td>
		<td style="text-align: center;">90.55</td>
		<td style="text-align: center;"> - </td>
		<td style="text-align: center;"> - </td>
		<td style="text-align: center;"> - </td>
		<td style="text-align: center;"> - </td>
		<td style="text-align: center;"> - </td>
	</tr>
</table>

The following figure shows the breakdown of the VTAB tasks:
![](assets/claim1.png)

<small>The models used are not available to the public and cannot be reproduced, as they are trained on the JFT-300M dataset, which is a private dataset owned by Google.</small>

To test the claim that vision transformers outperform convolutional neural networks on image classification tasks, we need to fine-tune the pretrained models on various datasets and compare their test accuracy. However, we face a challenge: we cannot access the pretrained models or the **JFT-300M** dataset that the authors used for pretraining. This is a private dataset that only they have. The only model we can use from the previous table is the **Vit-L/16 (I21k)**, which is marked in green. This model was pretrained on the **ImageNet-21k (I21k)** dataset and released by Google.

**How can we overcome this challenge and verify their claim**⁉️

A possible solution is to use other published models that were also pretrained on the **ImageNet-21k** dataset, such as the **BiT-L (I21k)** model. We can fine-tune these models on the same classification datasets as the vision transformer model and compare their performance. We can create a table like the following and fill the missing parts:

<table style="width: 100%;">
	<tr>
		<th style="text-align: center;">Model</th>
		<th style="text-align: center;">ImageNet</th>
		<th style="text-align: center;">ImageNet ReaL</th>
		<th style="text-align: center;">CIFAR-10</th>
		<th style="text-align: center;">CIFAR-100</th>
		<th style="text-align: center;">Oxford-IIIT Pets</th>
		<th style="text-align: center;">Oxford Flowers-102</th>
		<th style="text-align: center;">VTAB (19 tasks)</th>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-L/16 (I21k)</td>
		<td style="text-align: center;">85.30</td>
		<td style="text-align: center;">88.62</td>
		<td style="text-align: center;">99.15</td>
		<td style="text-align: center;">93.25</td>
		<td style="text-align: center;">94.67</td>
		<td style="text-align: center;">99.61</td>
		<td style="text-align: center;">72.72</td>
	</tr>
	<tr>
		<td style="text-align: center;">BiT-L (I21k)</td>
		<td style="text-align: center;">?</td>
		<td style="text-align: center;">?</td>
		<td style="text-align: center;">?</td>
		<td style="text-align: center;">?</td>
		<td style="text-align: center;">?</td>
		<td style="text-align: center;">?</td>
		<td style="text-align: center;">?</td>
	</tr>

</table>

***
:::

::: {.cell .markdown}
## Claim 2: The performance of the Vision Transformer on the classification task after fine tuning improves with the increase of the pretraining dataset size

The authors claim that their **Vision Transformer** models can learn more effectively from larger datasets than conventional **CNNs**, which enhances their performance on downstream tasks. This means that the vision transformer can transfer the knowledge learned from previous datasets to new ones more effectively than the **ResNet** models.

To demonstrate their claim, the authors compared the ResNet models and the vision transformer models that were pretrained on three different datasets: **ImageNet**, **ImageNet-21k** and **JFT-300M**. They then fine-tuned the models on the ImageNet dataset for classification. The figure below shows how the pretraining dataset size affects the test accuracy of the models.

![](assets/claim2.png)


To evaluate the performance of the vision transformer model, the author fine-tuned it on various datasets and presented the results in the tables below. The pretrained models that are marked in green are publicly available.

Model pretrained on the **ImageNet** dataset:

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

Model pretrained on the **ImageNet-21k** dataset:

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

Model pretrained on the **JFT-300M** dataset:

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



We want to test the qualitative claim by fine-tuning the pretrained models to classify images in the ImageNet dataset as the authors did and examine how the size of each pretraining dataset influences the final test accuracy. However, we cannot verify the results of the **JFT-300M** pretrained models as they are not publicly available.

We can only test the quantitative claims for the models pretrained on **ImageNet-21k**, since the models and dataset for **JFT-300M** are not accessible to the public. The models pretrained on **ImageNet-1k** are available, but they use a different optimizer (SAM) than the one described in the paper. This means that we might get similar results, but not exactly the same as the paper.

***
:::

::: {.cell .markdown}
## Claim 3: The hybrid Vision Transformer can perform better than both baseline and Vision Transformer after fine tuning it to different classification task

The authors propose a **hybrid vision transformer** that combines a **ResNet** backbone with a **vision transformer**, and claim that this model outperforms both the *pure vision transformer* and the pure *CNN* models. They argue that the ResNet layer provides a better feature representation for the transformer, enabling it to learn more information from the images.

To support their claim, the authors conduct various experiments using different models pretrained on the **JFT-300M** dataset and then fine-tuned on classification tasks on different datasets. They also vary the number of epochs as a hyperparameter and test the sensitivity of their results to this factor. The tables below show the results they obtained for each type of model.

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

We are unable to verify the quantitative results of the previous claim because we do not have access to the pretrained models or the **JFT-300M** dataset, which is a private dataset that the authors used for pretraining.

We want to test the claim that the **hybrid vision transformer** models can outperform both the **CNN** and **vision transformer** models on different classification tasks. However, we do not have the models or the data that the authors used for pretraining. To address this issue, we will use the models that the authors pretrained on the **ImageNet-21k** dataset, which is a public dataset. We will fine-tune these models on various classification tasks, such as CIFAR-10, CIFAR-100, Pets, and Flowers. Then, we will compare the test accuracy of each type of model: pure vision transformer, pure CNN, and hybrid vision transformer.

***
:::

::: {.cell .markdown}
**Note: There are other claims about the computational improvement of the vision transformer model compared to the traditional CNNs that achieve similar results. We will not address any of these claims due to their high computational cost. Moreover, some of these results are not reproducible because of the unavailability of the JFT-300M dataset.**

**Can you identify one of these claims and provide the experiment they did to support their claim? 🧐**

***
:::
