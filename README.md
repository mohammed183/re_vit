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

## Installation

To run the notebook, you will need to install the following dependencies:

- Python 3.6 or higher
- PyTorch 1.7 or higher
- NumPy
- Matplotlib
- Jupyter Notebook or Jupyter Lab
- torch
- torchvision
- transformers
- gdown

You can install them using pip or conda, for example:

```
pip install -r requirements.txt
```

or

```
conda install --file requirements.txt
```
or to avoid being prompt for yes for every package
```
conda install --yes --file requirements.txt
```

## Usage

To run the notebook, you can clone this repository and launch Jupyter notebook from the project directory:

```
git clone https://github.com/ucsc-ospo/re_vit.git
cd re_vit
make
jupyter notebook
```

Then, open the notebook file `Start_Here.ipynb` in the notebooks folder and follow the instructions.

Alternatively, you can use Google Colab to run the notebook online without installing anything. Just click on this link: <a target="_blank" href="https://colab.research.google.com/github/mohammed183/re_vit/blob/main/Start_Here.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.