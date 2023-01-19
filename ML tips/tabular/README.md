## Tabular data

### EDA

don't do too much EDA at the beginning, risk overfitting. Do "ml driven EDA"

### Working with dates

Use fastai [add_datepart](https://docs.fast.ai/tabular.transform.html#add_datepart) or [add_cyclic_datepart](https://docs.fast.ai/tabular.transform.html#add_cyclic_datepart)

### Saving dfs
save to feather `df.to_feather('...')`

### Speeding up experimentation/exploration

- use smaller subset of data
- use smaller rf (less trees)
- set_rf_sample() to let rf use subsets of data when fitting
- use full model/data at the last step

### Importance of a good validation set
- validation set needs to be representative of the test/deployment data (70/15/15 split is a good starting pt)
- can use test data to see how correlated is validation performance to test performance (only time to look at test data)
  1. build 5 models (varying in how good they perform on validation data)
  2. predict on both validation and test dataset
  3. plot val_score vs test_score, see how well is the correlation
- if there is a temporal aspect to the data, ALWAYS split by time
  - after getting a good model on validation data, retrain the same model (same hyperparameters) on train + val data (**for temporal data**)

### Look at feature importance ASAP
- build a rf/gbm (doesn't have to be very accurate), then evaluate feature importance right after
- use either sklearn or SHAP
- try throwing away unimportant columns and refit a model -> should get similar (or slightly) better results, but much faster
  - re-run feature importance since colinearity is removed, makes feature importance a lot more clearer
  
### One hot encoding
- can be useful for low cardinality categorical variables (6-7 can be a good starting point)
- use [`proc_df`](https://github.com/fastai/fastai/blob/master/old/fastai/structured.py) function from old fastai structured
- may not improve performance, but can yield additional insight into feature importance

### Removing redundant features
- use dendrograms (ON ONLY THE INTERESTING FEATURES IF YOU DID FEATURE IMPORTANCE BEFOREHAND)
- REMOVE REDUNDANT FEATURES AFTER FEATURE IMPORTANCE
```
cluster_columns(features) # new fastai2 feature
```
- the further the splits are to the bottom (or right) of the plot means they are more closely related
- Drop columns one at a time and see if validation score improves or drops
  - if only drops a little bit, but makes model simpler, can go with that option (tradeoff a bit of performance for speed)
  - Don't drop all columns in a group, even if dropping each column individually doesn't affect performance much
  
### Partial dependence plots
- looking at relationship between feature and label when all other features are the same
- more useful for understanding data, rather than for predictive power
```
from pdpbox import pdp
from plotnine import *
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, feat) # m here is the trained rf model, x is the x_train or sample of x_train
    return pdp.pdp_plot(p, feat_name, plot_lines=True, 
                        cluster=clusters is not None, 
                        n_cluster_centers=clusters)plot_pdp('YearMade')
                        
# then
plot_pdp('YearMade', clusters=5) 

# interaction plots between 2 features
feats = ['saleElapsed', 'YearMade']
p = pdp.pdp_interact(m, x, feats)
pdp.pdp_interact_plot(p, feats)
```

### Extrapolation for time dependent data (if test/live data is time dependent)

Only for tree based models (issues with extrapolation)

Remove time dependent variables from the model
1. Create label for `is_test = 1` or `is_test = 0`
2. Train model to predict `is_test`
3. Look at feature importance to see which features are most time sensitive
4. Remove each feature **one at a time** to see if improves performance on validation data
5. Remove the unhelpful features

Alternative: use NNs (can easily handle extrapolation into future) **OR** detrend data (with differencing)

### NN Categorical embedding with high cardinality variable

- Don't want to have too many `cat_vars` with high cardinality, will take up lots of parameters in embedding layer (since each level needs its own embedding layer)
- Use RF to see if you can remove any of those vars without degreading performance

### General Procedure
1. start with RF for steps above
2. Move on to GBT/NN after feature engineering

## Modelling options
https://twitter.com/marktenenholtz/status/1490671701884952576?t=f-HLbRhzn2g5uD6rsZnp1A&s=09&fbclid=IwAR1nNEhVXfdxWB9VXORJHpBsgflUKNay7IciBEl-IsSxSdSP6rhffBIrWsw

1. RF/Extratrees
2. GBM: lightgbm/catboost/xgboost
3. MLP: eg tabular learner in fastai
4. 1D-CNNs: https://www.kaggle.com/c/lish-moa/discussion/202256
5. TabNet 

### Tabnet
https://walkwithfastai.com/TabNet

### Shap + Fastai

https://github.com/muellerzr/fastai2-SHAP 

### imputing missing features with MICE
[MICE](https://github.com/AnotherSamWilson/miceForest)

### AutoML options for tabular models
[AutoGluon](https://auto.gluon.ai/stable/index.html)
[MLJar](https://github.com/mljar/mljar-supervised)

### Speeding up tabular learning by converting DF -> numpy
https://muellerzr.github.io/fastblog/2020/04/22/TabularNumpy.html

### SAINT 
https://twitter.com/TheZachMueller/status/1400533460784197638

### GBM
- light gbm, catboost, xgboost
- https://catboost.ai/docs/concepts/about.html
- hyperparam tuning for catboost https://catboost.ai/docs/concepts/parameter-tuning.html
- hyperparam tuning xgboost + optuna: https://github.com/tunguz/TPS_11_2021/blob/main/scripts/XGB_Optuna_Dask_DGX_Station_A100_3.ipynb

### DeepETA: How Uber Predicts Arrival Times Using Deep Learning 
https://eng.uber.com/deepeta-how-uber-predicts-arrival-times/

- transformer model > xgboost at scale

### On Embeddings for Numerical Features in Tabular Deep Learning
https://arxiv.org/abs/2203.05556

- tl;dr: embeddings for continuous features as well -> gainz

### entity embeddings in sklearn
https://github.com/cpa-analytics/embedding-encoder


### Categorical feature encodings library
https://github.com/scikit-learn-contrib/category_encoders

http://contrib.scikit-learn.org/category_encoders/index.html

### Sample kaggle tabular data pipeline 
https://github.com/arnabbiswas1/kaggle_pipeline_tps_aug_22

### (Regualrization technique) swap noise
- https://twitter.com/rasbt/status/1567521047221604353
- with some percentage (15%), replace values of a column with another value in the same column

### regularization techniques for trainning NN on tabular data
https://twitter.com/rasbt/status/1572616437977546754?fbclid=IwAR3olVyGHIOmQ6RwAq-vnkxXs-Dxvch46YFyYY3zBhtFoDZpPVMrAb7-hhg

### GBDT on residual(target - highly_correlated_feature) instead of target itself
- https://twitter.com/a_erdem4/status/1587791119772811265
- start to minimize from a smaller error
- keep the feature in the GBDT model

### tuning xgboost/lightgbm
- 99% of time only need to tune
```
• Objective function
• num_leaves/max_depth
• feature_fraction
• bagging_fraction
• min_child_samples

```
- usually default lr of 0.05 is fine, but can also tune

### uplift modelling
https://medium.com/@rndonnelly/a-simpler-alternative-to-x-learner-for-uplift-modeling-f3a11ebf6bf1
