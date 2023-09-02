::: {.cell .markdown}
# Claims 📝

The paper “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” evaluates the performance of **ResNet**, **ViT**, and **hybrid** models on various image classification tasks and makes several claims based on their assessment. In this section, we present these claims and propose a way to test them using the **pre-trained** models provided by the authors. The table below, taken from the paper, shows the different variants of the models used by the authors, who also use a specific notation to refer to the models. For example, “ViT-L/16” refers to the “Large” variant with an input patch size of 16x16. The authors also use an improved ResNet as a baseline model, referred to as [“ResNet (BiT)”](https://arxiv.org/abs/1912.11370), which has published models available on [GitHub](https://github.com/google-research/big_transfer).

| Model     | Layers | Hidden size D | MLP size | Heads | Params |
| :-------: | :----: | :-----------: | :------: | :---: | :----: |
| ViT-Base  | 12     | 768           | 3072     | 12    | 86M    |
| ViT-Large | 24     | 1024          | 4096     | 16    | 307M   |
| ViT-Huge  | 32     | 1280          | 5120     | 16    | 632M   |


We aim to evaluate the claims made in the Vision Transformer paper both qualitatively and quantitatively. However, we face several **challenges** in doing so: 

- Some of the **pre-trained** models used by the authors are not publicly available

- We do not have sufficient **computational** resources to train the models from scratch

- Some of the datasets used for pretraining are private and inaccessible

As a result, we may not be able to reproduce all of the results presented in the paper. We will carefully examine each claim to determine which parts can be reproduced and which cannot be reproduced given the challenges associated with that claim.

***
:::

::: {.cell .markdown}
## Primary Claim: Vision Transformer outperforms state of the art CNNs on various classification tasks after pretraining on large datasets

This claim suggests that the vision transformer can leverage more knowledge from large datasets than CNN models during pretraining and transfer this knowledge to the fine-tuning task. This implies that the vision transformer can achieve higher or comparable accuracies to the state of the art models on different classification tasks.

The authors support their claim by pretraining three versions of the vision transformer and evaluating their performance on various benchmarks. However, they use the **JFT-300M** dataset for pretraining, which is a private dataset owned by Google and *not available* to the public. Additionally, the authors do not share the models pre-trained on the **JFT-300M** dataset. The only explicit comparison presented in Table 2 of the paper is between Vision Transformers and CNNs pre-trained on the same private **JFT-300M** dataset. The authors also show results for a Vision Transformer pre-trained on the publicly available **ImageNet-21k** dataset but not compared with a CNN pre-trained on the same dataset. It is worth noting that *BiT-L is a ResNet152x4 model*, which is a type of CNN. The following table shows a subset of the results mentioned above:

<table style="width: 100%;">
	<tr>
		<th style="text-align: center;">Model</th>
		<th style="text-align: center;">ImageNet</th>
		<th style="text-align: center;">ImageNet ReaL</th>
		<th style="text-align: center;">CIFAR-10</th>
		<th style="text-align: center;">CIFAR-100</th>
		<th style="text-align: center;">Oxford-IIIT Pets</th>
		<th style="text-align: center;">Oxford Flowers-102</th>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-H/14 (JFT)</td>
		<td style="text-align: center;">88.55</td>
		<td style="text-align: center;">90.72</td>
		<td style="text-align: center;">99.50</td>
		<td style="text-align: center;">94.55</td>
		<td style="text-align: center;">97.56</td>
		<td style="text-align: center;">99.68</td>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-L/16 (JFT)</td>
		<td style="text-align: center;">87.76</td>
		<td style="text-align: center;">90.54</td>
		<td style="text-align: center;">99.42</td>
		<td style="text-align: center;">93.90</td>
		<td style="text-align: center;">97.32</td>
		<td style="text-align: center;">99.74</td>
	</tr>
	<tr style="background-color: lightgreen;">
		<td style="text-align: center;">ViT-L/16 (I21k)</td>
		<td style="text-align: center;">85.30</td>
		<td style="text-align: center;">88.62</td>
		<td style="text-align: center;">99.15</td>
		<td style="text-align: center;">93.25</td>
		<td style="text-align: center;">94.67</td>
		<td style="text-align: center;">99.61</td>
	</tr>
	<tr>
		<td style="text-align: center;">BiT-L (JFT)</td>
		<td style="text-align: center;">87.54</td>
		<td style="text-align: center;">90.54</td>
		<td style="text-align: center;">99.37</td>
		<td style="text-align: center;">93.51</td>
		<td style="text-align: center;">96.62</td>
		<td style="text-align: center;">99.63</td>
	</tr>
	<tr style=“white-space: nowrap;”>
		<td style="text-align: center;">Noisy Student</td>
		<td style="text-align: center;">88.4</td>
		<td style="text-align: center;">90.55</td>
		<td style="text-align: center;"> - </td>
		<td style="text-align: center;"> - </td>
		<td style="text-align: center;"> - </td>
		<td style="text-align: center;"> - </td>
	</tr>
</table>

To test the claim that vision transformers outperform convolutional neural networks on image classification tasks, we need to fine-tune the pre-trained models on various datasets and compare their test accuracy. However, we face a challenge: we cannot access the pre-trained models or the **JFT-300M** dataset that the authors used for pretraining. This is a private dataset that only they have. The only model we can use from the previous table is the **Vit-L/16 (I21k)**, which is marked in green. This model was pre-trained on the **ImageNet-21k (I21k)** dataset and released by Google.

**How can we overcome this challenge and verify their claim**⁉️

One possible solution to evaluate the performance of the Vision Transformer model is to use the pre-trained model on the **ImageNet-21k** dataset, which the authors have published results for, and compare it with the same CNN (BiT-L) but pre-trained on the same dataset which is we will refer to as BiT-M (provided by authors). We can fine-tune these models on the same classification datasets as the vision transformer model and compare their performance. We can then fine-tune the BiT-M on **ImageNet-1k**, **CIFAR-10**, **CIFAR-100**, **Oxford-IIIT Pets** and **Oxford Flowers-102** and use it as a baseline. Then fine-tune the ViT-L/16 (I21k) on the same datasets and compare the results with the baseline model. We can then create a table similar to the following with our results and we can see which quantitative results from the paper we are able to reproduce.

<table style="width: 100%;">
	<tr>
		<th style="text-align: center;">Model</th>
		<th style="text-align: center;">ImageNet</th>
		<th style="text-align: center;">ImageNet ReaL</th>
		<th style="text-align: center;">CIFAR-10</th>
		<th style="text-align: center;">CIFAR-100</th>
		<th style="text-align: center;">Oxford-IIIT Pets</th>
		<th style="text-align: center;">Oxford Flowers-102</th>
	</tr>
	<tr>
		<td style="text-align: center;">ViT-L/16 (I21k)</td>
		<td style="text-align: center;">85.30</td>
		<td style="text-align: center;">88.62</td>
		<td style="text-align: center;">99.15</td>
		<td style="text-align: center;">93.25</td>
		<td style="text-align: center;">94.67</td>
		<td style="text-align: center;">99.61</td>
	</tr>
	<tr>
		<td style="text-align: center;">BiT-M (I21k)</td>
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
**Based on the information provided, can you explain which parts of the authors’ primary claim can and cannot be evaluated? Additionally, can you describe how can we evaluate the claims that are possible to evaluate? 🤔**

***
:::

::: {.cell .markdown}
**🛑 There are other claims about the computational improvement of the vision transformer model compared to the traditional CNNs that achieve similar results. We will not address any of these claims due to their high computational cost. Moreover, some of these results are not reproducible because of the unavailability of the JFT-300M dataset.**

**Can you identify one of these claims and provide the experiment they did to support their claim? 🧐**

***
:::

::: {.cell .markdown}
Other claims about the performance of the Vision Transformer can be found in the [Going Further notebook](05-Going_Further.ipynb) with the experiments used to verify these claims and explanation for the challenges that we might face while verifying them.

***
:::
