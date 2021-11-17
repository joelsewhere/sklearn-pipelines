# Creating a Custom Sklearn Pipeline
> How to include custom data preprocessing steps in an sklearn pipeline. 

In this notebook we will import the [income classification dataset](https://www.kaggle.com/lodetomasi1995/income-classification/data), review common preprocessing steps, and then introduce how those steps can be included in an sklearn pipeline. 

![](https://media.giphy.com/media/Jwp4sxM0Rjk1W/giphy.gif)


```python
# Standard Imports
import pandas as pd
import numpy as np

# Transformers
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Modeling Evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Pipelines
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

# Machine Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```


```python
df = pd.read_csv('data/train.csv')
df.head()
```

## Develop a preprocessing strategy


### Task 1

Write a function called `bin_middle_age` that can be applied to the `age` column in `X_train` and returns a 1 if the age is 45-64 and a zero for every other age. 

### Task 2

Write a function called `bin_capital` that can be applied to the `capital-gain` and `capital-loss` columns in `X_train` and returns a 1 if the input is more than zero and a 0 for anything else.

### Task 3

Please write code to fit a one hot encoder to all of the object datatypes. Transform the object columns in `X_train` and turn them into a dataframe. For this final step, I'll give you two clues: "sparse" and "dense". Only one of them will be needed.

### Task 4

Please write code to scale the `'hours-per-week'` column in `X_train'.

### Task 5
Merge the transformed features into a new dataframe called `modeling_df`.


```python
X_train, X_test, y_train, y_test = train_test_split(df.drop('income', axis = 1), 
                                                    df.income,
                                                    random_state = 2020)
X_train.reset_index(drop=True, inplace=True)

# Task 1
# ===========================================
def bin_middle_age(age):
    pass

X_train['age'] = X_train.age.apply(bin_middle_age)
# ===========================================

# Task 2
# ===========================================
def bin_capital(x):
    pass

X_train['capital-gain'] = X_train['capital-gain'].apply(bin_capital)
X_train['capital-loss'] = X_train['capital-loss'].apply(bin_capital)

 
X_train.reset_index(drop=True, inplace=True)
# ===========================================

# Task 3
# ===========================================
hot_encoder = None
categoricals = None
# ===========================================

# Task 4
# ===========================================
hours_scaler = None
hours_per_week = None
# ===========================================

# Task 5
# ===========================================
modeling_df = None
# ===========================================
```

# Move all of this into a Pipeline


### Writing a custom transformer

Above we used two sklearn transformers and two custom functions to format our dataframe. This means, that we will need to create two custom transformers. The sklearn transformers can be used as they are.

To do this, we will create a class called `BinAge` that inherits from the sklearn classes, `TransformerMixin` and `BaseEstimator`. This class should have the following methods:

1. `fit`
    - This method should have three arguments
        1. self
        2. `X`.
        3. `y=None`
    - This method should return `self`.
    
1. `_bin_data`
    - This method is our function for binning the age column
    
1. `transform`
    - This method should have two arguments
        1. self
        2. `X`
    - This method should apply the `_bin_data` method to `X`
    - Return the binned data


```python
from numpy import vectorize

class BinAge():
    
    def fit(self, X, y=None):
        pass
    
    @vectorize
    def _bin_data(x):
        return int(x >= 45 and x <= 64)
    
    def transform(self, X):
        pass
```

**Now repeat the process for a `BinCapital` Transformer!**


```python
class BinCapital():
    pass
```

## Create pipeline

To make this pipeline, we will use the following sklearn functions:

1. `make_column_transformer`
> This function receives "Tuples of the form `(transformer, [columns])` specifying the transformer objects to be applied to subsets of the data."
2. `make_column_selector`
> "Selects columns based on datatype or the columns name with a regex. When using multiple selection criteria, all criteria must match for a column to be selected."
3. `make_pipeline`
> Used to create a pipeline of inputer transformer and estimator objects.


```python
preprocessing = make_column_transformer((BinAge(), ['age']),
                                      (BinCapital(), ['capital-gain']),
                                      (BinCapital(), ['capital-loss']),
                                      (OneHotEncoder(sparse=False, handle_unknown='ignore'),
                                       make_column_selector(dtype_include=object)),
                                      (StandardScaler(), ['hours-per-week']),
                                      remainder='drop')
```

Now all of our preprocessing can be done with the `fit_transform` method!


```python
preprocessing.fit_transform(X_train)
```




    array([[ 0.        ,  0.        ,  1.        , ...,  0.        ,
             0.        , -2.87747181],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.77531045],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.20709987],
           ...,
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        , -0.03641894],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.28827281],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.77531045]])




```python
#__SOLUTION__
preprocessing.fit_transform(X_train)
```




    array([[ 0.        ,  0.        ,  1.        , ...,  0.        ,
             0.        , -2.87747181],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.77531045],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.20709987],
           ...,
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        , -0.03641894],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.28827281],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.77531045]])



To finish up pipeline, we can add a machine learning model to a new pipeline!


```python
dt_pipeline = make_pipeline(preprocessing, DecisionTreeClassifier())
rf_pipeline = make_pipeline(preprocessing, RandomForestClassifier(max_depth=10))
```

## Our pipelines are built!

Now we can run them through cross validation!


```python
cross_val_score(dt_pipeline, X_train, y_train)
```




    array([0.80727934, 0.81146497, 0.80746133, 0.81234074, 0.80269385])




```python
#__SOLUTION__
cross_val_score(dt_pipeline, X_train, y_train)
```




    array([0.80527753, 0.81091902, 0.80709736, 0.8119767 , 0.804332  ])




```python
cross_val_score(rf_pipeline, X_train, y_train)
```




    array([0.83676069, 0.833303  , 0.83130118, 0.8380051 , 0.82963233])




```python
#__SOLUTION__
cross_val_score(rf_pipeline, X_train, y_train)
```




    array([0.83676069, 0.83275705, 0.83093722, 0.83946123, 0.83199854])




```python
rf_pipeline.fit(X_train, y_train)
train_preds = rf_pipeline.predict(X_train)
val_preds = rf_pipeline.predict(X_val)
print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
print(f'Validation Accuracy: {accuracy_score(y_val, val_preds)}')
```

    Training Accuracy: 0.8435554908455575
    Validation Accuracy: 0.840467350949989



```python
#__SOLUTION__
rf_pipeline.fit(X_train, y_train)
train_preds = rf_pipeline.predict(X_train)
val_preds = rf_pipeline.predict(X_val)
print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
print(f'Validation Accuracy: {accuracy_score(y_val, val_preds)}')
```

    Training Accuracy: 0.8427183052451498
    Validation Accuracy: 0.8376283031229526


Finally, we can fit the final pipeline on all of the data and test it on an additional hold out set!


```python
rf_pipeline.fit(df.drop('income', axis = 1), df.income)
```




    Pipeline(steps=[('columntransformer',
                     ColumnTransformer(transformers=[('binage', BinAge(), ['age']),
                                                     ('bincapital-1', BinCapital(),
                                                      ['capital-gain']),
                                                     ('bincapital-2', BinCapital(),
                                                      ['capital-loss']),
                                                     ('onehotencoder',
                                                      OneHotEncoder(handle_unknown='ignore',
                                                                    sparse=False),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x7fdfc1fd9590>),
                                                     ('standardscaler',
                                                      StandardScaler(),
                                                      ['hours-per-week'])])),
                    ('randomforestclassifier',
                     RandomForestClassifier(max_depth=10))])




```python
#__SOLUTION__
rf_pipeline.fit(df.drop('income', axis = 1), df.income)
```




    Pipeline(steps=[('columntransformer',
                     ColumnTransformer(transformers=[('binage', BinAge(), ['age']),
                                                     ('bincapital-1', BinCapital(),
                                                      ['capital-gain']),
                                                     ('bincapital-2', BinCapital(),
                                                      ['capital-loss']),
                                                     ('onehotencoder',
                                                      OneHotEncoder(handle_unknown='ignore',
                                                                    sparse=False),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x7fdfc1fd9590>),
                                                     ('standardscaler',
                                                      StandardScaler(),
                                                      ['hours-per-week'])])),
                    ('randomforestclassifier',
                     RandomForestClassifier(max_depth=10))])



Load in the hold out set and make predictions!


```python
# Import holdout data
test = pd.read_csv('data/test.csv')
# Seperate features from the target
X_test, y_test = test.drop(columns=['income']), test.income
# Score the model
rf_pipeline.score(X_test, y_test)
```




    0.8423552534599951




```python
#__SOLUTION__
# Import holdout data
test = pd.read_csv('data/test.csv')
# Seperate features from the target
X_test, y_test = test.drop(columns=['income']), test.income
# Score the model
rf_pipeline.score(X_test, y_test)
```




    0.8423552534599951



### Save the model to disk


```python
# Merge training and hold out sets
full_data = pd.concat([df, test])

# Seperate the features from the target
X, y = df.drop(columns=['income']), df.income

# Fit the model to *all* observations
rf_pipeline.fit(X, y)

# Save the fit model to disk
file = open('model_v1.pkl', 'wb')
pickle.dump(rf_pipeline, file)
file.close()
```

### Check the saved model works when loaded


```python
# Load the model
file = open('model_v1.pkl', 'rb')
model = pickle.load(file)
file.close()

# Generate predictions
model.predict(X)
```




    array(['>50K', '<=50K', '<=50K', ..., '<=50K', '<=50K', '<=50K'],
          dtype=object)




```python
#__SOLUTION__
# Load the model
file = open('model_v1.pkl', 'rb')
model = pickle.load(file)
file.close()

# Generate predictions
model.predict(X)
```




    array(['>50K', '<=50K', '<=50K', ..., '<=50K', '<=50K', '<=50K'],
          dtype=object)



### Visualize the pipeline!


```python
from sklearn import set_config

set_config(display="diagram")
model
```




<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c7965859-c316-4852-9687-dd580d122b7b" type="checkbox" ><label class="sk-toggleable__label" for="c7965859-c316-4852-9687-dd580d122b7b">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('columntransformer',
                 ColumnTransformer(transformers=[('binage', BinAge(), ['age']),
                                                 ('bincapital-1', BinCapital(),
                                                  ['capital-gain']),
                                                 ('bincapital-2', BinCapital(),
                                                  ['capital-loss']),
                                                 ('onehotencoder',
                                                  OneHotEncoder(handle_unknown='ignore',
                                                                sparse=False),
                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7fdfad207bd0>),
                                                 ('standardscaler',
                                                  StandardScaler(),
                                                  ['hours-per-week'])])),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=10))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d2e7c754-8489-41fd-901b-5f29edeae048" type="checkbox" ><label class="sk-toggleable__label" for="d2e7c754-8489-41fd-901b-5f29edeae048">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[('binage', BinAge(), ['age']),
                                ('bincapital-1', BinCapital(),
                                 ['capital-gain']),
                                ('bincapital-2', BinCapital(),
                                 ['capital-loss']),
                                ('onehotencoder',
                                 OneHotEncoder(handle_unknown='ignore',
                                               sparse=False),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7fdfad207bd0>),
                                ('standardscaler', StandardScaler(),
                                 ['hours-per-week'])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e8dbeacd-2a43-4667-bd56-46b9f2357d8a" type="checkbox" ><label class="sk-toggleable__label" for="e8dbeacd-2a43-4667-bd56-46b9f2357d8a">binage</label><div class="sk-toggleable__content"><pre>['age']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2c99c595-24aa-4748-93cb-780004207380" type="checkbox" ><label class="sk-toggleable__label" for="2c99c595-24aa-4748-93cb-780004207380">BinAge</label><div class="sk-toggleable__content"><pre>BinAge()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a117ee16-174b-484d-9de1-5ec05b1eaabc" type="checkbox" ><label class="sk-toggleable__label" for="a117ee16-174b-484d-9de1-5ec05b1eaabc">bincapital-1</label><div class="sk-toggleable__content"><pre>['capital-gain']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d9247a84-137f-4cac-b76d-0ff5c7100288" type="checkbox" ><label class="sk-toggleable__label" for="d9247a84-137f-4cac-b76d-0ff5c7100288">BinCapital</label><div class="sk-toggleable__content"><pre>BinCapital()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="48d9ecc8-053c-4bdf-837d-33708018b3eb" type="checkbox" ><label class="sk-toggleable__label" for="48d9ecc8-053c-4bdf-837d-33708018b3eb">bincapital-2</label><div class="sk-toggleable__content"><pre>['capital-loss']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e64aba64-bef8-40f3-ba82-d24991052dc2" type="checkbox" ><label class="sk-toggleable__label" for="e64aba64-bef8-40f3-ba82-d24991052dc2">BinCapital</label><div class="sk-toggleable__content"><pre>BinCapital()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ffcc5d5f-d51f-49d1-9ccf-9326971fa0c2" type="checkbox" ><label class="sk-toggleable__label" for="ffcc5d5f-d51f-49d1-9ccf-9326971fa0c2">onehotencoder</label><div class="sk-toggleable__content"><pre><sklearn.compose._column_transformer.make_column_selector object at 0x7fdfad207bd0></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a285c11d-bdea-418f-b688-804f45a7bc54" type="checkbox" ><label class="sk-toggleable__label" for="a285c11d-bdea-418f-b688-804f45a7bc54">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown='ignore', sparse=False)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="692cb28c-e792-436b-9aea-b7c7586adda4" type="checkbox" ><label class="sk-toggleable__label" for="692cb28c-e792-436b-9aea-b7c7586adda4">standardscaler</label><div class="sk-toggleable__content"><pre>['hours-per-week']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="672d8757-29ab-4160-89eb-f128575348eb" type="checkbox" ><label class="sk-toggleable__label" for="672d8757-29ab-4160-89eb-f128575348eb">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ea6d6b8d-bc01-48cb-b8e3-641eaad77b1c" type="checkbox" ><label class="sk-toggleable__label" for="ea6d6b8d-bc01-48cb-b8e3-641eaad77b1c">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_depth=10)</pre></div></div></div></div></div></div></div>




```python
#__SOLUTION__
from sklearn import set_config

set_config(display="diagram")
model
```
