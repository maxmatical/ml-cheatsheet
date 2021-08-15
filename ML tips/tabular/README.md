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
- light gbm or catboost (catboost seems to be best single model)
- https://catboost.ai/docs/concepts/about.html
- hyperparam tuning for catboost https://catboost.ai/docs/concepts/parameter-tuning.html
