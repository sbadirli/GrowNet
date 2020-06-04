import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_svmlight_file

# load data
tr_x, tr_y = load_svmlight_file('./higgs.train')
te_x, te_y = load_svmlight_file('./higgs.test')

# grid search
param = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'learning_rate': 0.05,
    'n_estimators': 800,
    'max_depth': 7,
    'reg_lambda': 0.02,
}

# regressor
model = xgb.XGBRegressor(verbosity=2, seed=0, **param)
model.fit(tr_x, tr_y)

# predict on test data
auc = roc_auc_score(te_y, model.predict(te_x))
print(auc)
