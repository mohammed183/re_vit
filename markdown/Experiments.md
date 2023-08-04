::: {.cell .markdown}
# Experiments

In this section, we will attempt to verify the qualitative and quantitative aspects of each claim. We will indicate which claims cannot be verified due to the lack of the material published by the authors. We will mainly use pretrained models published to verify these claims. Below is a table with all the models mentioned in the paper and which versions of it are publicly available.

| Model          | Pretrained ImageNet | pretrained ImageNet-21 | Pretrained JFT |
| :------------: | :-----------------: | :--------------------: | :------------: |
| ResNet50x1     | Yes                 | Yes                    | No             |
| ResNet50x2     | No                  | No                     | No             |
| ResNet101x1    | Yes                 | Yes                    | No             |
| ResNet152x1    | No                  | No                     | No             |
| ResNet152x2    | Yes                 | Yes                    | No             |
| ResNet152x4    | Yes                 | Yes                    | No             |
| ResNet200x3    | No                  | No                     | No             |
| ViT-B/32       | Yes (SAM)           | Yes                    | No             |
| ViT-B/16       | Yes (SAM)           | Yes                    | No             |
| ViT-L/32       | Yes (SAM)           | Yes                    | No             |
| ViT-L/16       | Yes (SAM)           | Yes                    | No             |
| ViT-H/14       | No                  | Yes                    | No             |
| R50x1+ViT-B/32 | No                  | No (but R26 available) | No             |
| R50x1+ViT-B/16 | No                  | Yes                    | No             |
| R50x1+ViT-L/32 | No                  | Yes                    | No             |
| R50x1+ViT-L/16 | No                  | Yes                    | No             |

Our task will be to try to reproduce all the qualitative claims, as well as any possible quantitative claims using the pretrained models presented in the previous table.

***
:::

::: {.cell .markdown}
## Experiment 1:

In this experiment we want to reproduce the claim: "Vision Transformer outperforms state of the art CNNs on various classification tasks after pretraining on large datasets" by using the only available pretrained model in the table in that claim and compare it to the other model that are pretrained on the ImageNet-21k unlike in the original paper where the other models were trained on the JFT-300M private dataset.

***
:::

::: {.cell .markdown}
After running this experiment we should be able to fill the following table and then we validate the claim:

***
:::


::: {.cell .markdown}
This Experiment is split into two notebooks:

- The first notebook we train the baseline ResNet-152x4 model (Here will be link to notebook)
- The second notebook we train the Vit-L/16 model (Here will be link to notebook)
***
:::
