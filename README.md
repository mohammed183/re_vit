# [Re] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

This project is part of the [UCSC OSPO](https://ospo.ucsc.edu/) summer of reproducibility fellowship and aims to create an interactive notebook that can be used to teach undergraduate or graduate students different levels of reproducibility in computer vision research.

The project is based on the paper "[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)" by Dosovitskiy et al., which introduces a novel way of applying the transformer architecture, which was originally designed for natural language processing, to image recognition tasks. The paper shows that transformers can achieve state-of-the-art results on several image classification benchmarks, such as ImageNet, when trained on large-scale datasets.

The notebook will guide the students through the following steps:

- Introduce and explain the problem and the claims of the original paper
- Determine which claims can be tested using the available data and models
- Use the code and functions provided by the authors to test each claim
- Compare their results with the original paper and other baselines
- Evaluate the reproducibility of the research

The notebook will use Python and PyTorch as the main tools and will require some basic knowledge of computer vision and neural networks.

Here's a revised version of the text:

## Installation

To run the notebook, you'll need to install the following dependencies:

- Python 3.6 or higher
- PyTorch 1.7 or higher
- NumPy
- Matplotlib
- Jupyter Notebook or Jupyter Lab
- torch
- torchvision
- transformers
- gdown

There are three options for installing and running the notebook:

### Option 1: Run Locally on Your Device

To run the notebook locally on your device, you'll need to install Python and Jupyter Notebook. Once you have those installed, follow these steps:

1. Clone the repository and navigate to the `re_vit` directory by running the following command:
```
$ git clone https://github.com/mohammed183/re_vit.git && cd re_vit
```

2. Install the required packages by running this command:
```
$ pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Launch Jupyter Notebook by running this command:
```
$ jupyter notebook
```

4. In Jupyter Notebook, open the `Start_Here.ipynb` file located in the `notebooks` folder and follow the instructions.

Please note that you will need a 24GB GPU to run the ResNet notebook.

### Option 2: Run on Chameleon Cloud

You can run the notebook on Chameleon Cloud using either a Colab frontend or a Jupyter Lab frontend. To do this, follow these steps:

1. Clone the repository on the Jupyter interface for Chameleon Cloud by running the following command:
```
$ git clone https://github.com/mohammed183/re_vit.git && cd re_vit
```

2. Open the `Reserve.ipynb` notebook, which is available in the `re_vit` directory.

3. Follow the steps in the `Reserve.ipynb` notebook to reserve an instance on Chameleon Cloud and run it with your desired frontend.

Given the high resource requirements for the experiments in the notebooks, this option is likely to be the most suitable.

### Option 3: Run on Google Colab

You can also run the notebook on Google Colab. However, please note that the `ResNet.ipynb` notebook requires a high amount of GPU memory (24GB), which may not be available with a free Colab account. To open the `Start_Here.ipynb` file on Colab and navigate through the notebooks, click this button: <a target="_blank" href="https://colab.research.google.com/github/mohammed183/re_vit/blob/main/Start_Here.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.