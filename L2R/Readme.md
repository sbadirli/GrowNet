- Data loading and creating dataloader are handled in GrowNet/L2R/data/data.py. If you want to try new data please put it into Microsoft data format. 

- Individual model class and ensemble architecture are in GrowNet/L2R/models:  mlp.py and dynamic_net.py. 
You can increase number of hidden layers or change activation function from here: mlp.py

- train.sh contains pairwise-loss implementation. If you want to try I-divergence or MSE loss implementations just change the python -u main_l2r_pairwise_cv.py to python -u main_l2r_idiv_cv.py (or main_l2r_mse_cv.py). You can also change the dtaset to yahoo, but when you do, change the feature dimension as well (from 136 to 518). You may want to alter the hidden layer dimension as well, say 128 or 256.