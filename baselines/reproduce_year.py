import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# load data
tr_npz = np.load('./YearPredictionMSD_tr.npz')
te_npz = np.load('./YearPredictionMSD_te.npz')

# grid search
param = {
    'learning_rate': 0.05,
    'n_estimators': 800,
    'max_depth': 7,
    'reg_lambda': 0.02,
}

# regressor
model = xgb.XGBRegressor(objective='reg:squarederror',
                         verbosity=2,
                         seed=0,
                         **param)
model.fit(tr_npz['features'], tr_npz['labels'])

# predict on test data
mse = mean_squared_error(te_npz['labels'], model.predict(te_npz['features']))
print(np.sqrt(mse))
