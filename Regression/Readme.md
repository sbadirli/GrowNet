- Data loading and creating dataloader are handled in GrowNet/Regression/data/data.py. If you want to try new data please check the LibSVMRegdata function in data.py for the right format. 

- Individual model class and ensemble architecture are in GrowNet/Reg/models:  mlp.py and dynamic_net.py. 
You can increase number of hidden layers or change activation function from here: mlp.py

- train.sh will reproduce the results for Music Year Prediction data. You can change the dataset to slice_localization and feature dimension accordingly. You may also want to change hidden layre dimension to 128 or more for slice localization data.