::: {.cell .markdown}
# More on CNNs

In this notebook, we explore several convolutional neural networks related to those discussed in previous notebooks. Our goal is to demonstrate the advancements made in image classification using CNNs. To that end, we introduce two papers on convolutional neural networks below.

***
:::

::: {.cell .markdown}
## Big Transfer

The paper [“Big Transfer (BiT): General Visual Representation Learning” by Kolesnikov et al. (2020)](https://arxiv.org/abs/1912.11370) was published **before** the release of the Vision Transformer paper. Figure 1 from the paper, shown below, compares the performance of the **BiT** model to state-of-the-art models at the time on five datasets used for fine-tuning and comparison.

![](assets/BiT.png)

The authors of the **Vision Transformer** paper used the models described in the **BiT** paper to benchmark the performance of their Vision Transformers. In our ResNet notebook, we implemented some parts as described in the BiT paper, as the authors of the Vision Transformer paper did not provide complete details on their implementation of ResNet models. You can find their code in the [official repository](https://github.com/google-research/big_transfer) with the implementation of their models.

***
:::

::: {.cell .markdown}
## ConvNeXt

The paper [“A ConvNet for the 2020s” by Liu, Zhuang, et al. (2022)](https://arxiv.org/abs/1912.11370) was published **after** the release of the **Vision Transformer**. This paper proposes a convolutional neural network model called **ConvNeXt**, which outperforms the Vision Transformer on various datasets. Figure 1 from the paper, shown below, presents the results achieved by ConvNeXt.

![](assets/ConvNext.png)

You can verify the performance of ConvNeXt by accessing the [official code repository on GitHub](https://github.com/facebookresearch/ConvNeXt) and running the provided notebook. The repository contains a [Colab notebook](https://colab.research.google.com/drive/1CBYTIZ4tBMsVL5cqu9N_-Q3TBprqsfEO?usp=sharing) that allows you to test the model and compare its results to those that we got for the Vision Transformer.

***
:::
