# GrowNet

Original PyTorch implementation of "Gradient Boosting Neural Networks: GrowNet" 

Paper at: https://arxiv.org/pdf/2002.07971.pdf

<p align="center">
  <img width="800" src="Model.png">
</p>
<p align="justify">

## Getting Started

Little summary of GrowNet and some insight


## Prerequisites

The code was implemented in Python 3.6.10 and utilized the packages (full list) in requirements.txt file. The platform I used was linux-64. Most important packages you need are the followings:
```
cudatoolkit=10.1.243 
numpy=1.18.1 
pandas=1.0.0 
python=3.6.10 
pytorch=1.4.0 
```

## Installing

To run the code, You may create a conda environment (assuming you already have miniconda3 installed) by the following command on terminal:

```
conda create --name grownet --file requirements.txt
```

## Data

You can download the data used in the paper from Google OneDrive link. Please put it into data folder in GrowNet file next to Classification/L2R/Regression folders.

## Experiments

To reproduce the results from pape, first activate conda virtual environment

```
conda activate grownet
```
Then simply navigate to the task folder: Classification, L2R or Regression and execute the following command on terminal:

```
./train.sh
```

You may change the dataset, number of hidden layers, number of hidden units in hidden units, batch size, learning rate and etc from train.sh. 

The results may vary 1% or less between identical runs due to random initialization.

### Contact

Feel free to drop me an email if you have any questions: s.badirli@gmail.com

### Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
