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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>educational-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53</td>
      <td>Local-gov</td>
      <td>283602</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28</td>
      <td>Self-emp-not-inc</td>
      <td>35864</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>Other</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>Iran</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>Private</td>
      <td>146764</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Divorced</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49</td>
      <td>Private</td>
      <td>59380</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-spouse-absent</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51</td>
      <td>Self-emp-inc</td>
      <td>338836</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>



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
X_train, X_val, y_train, y_val = train_test_split(df.drop('income', axis = 1), 
                                                    df.income,
                                                    random_state = 2020)  
X_train.reset_index(drop=True, inplace=True)

# Task 1
# ===========================================
def bin_middle_age(age):
    return int(age >= 45 and age <= 64)

X_train['age'] = X_train.age.apply(bin_middle_age)
# ===========================================

# Task 2
# ===========================================
def bin_capital(x):
    return int(x > 0)

X_train['capital-gain'] = X_train['capital-gain'].apply(bin_capital)
X_train['capital-loss'] = X_train['capital-loss'].apply(bin_capital)
# ===========================================

# Task 3
# ===========================================
hot_encoder = OneHotEncoder(sparse=False)
categoricals = hot_encoder.fit_transform(X_train.select_dtypes(object))
categoricals = pd.DataFrame(categoricals, columns = hot_encoder.get_feature_names())
# ===========================================

# Task 4
# ===========================================
hours_scaler = StandardScaler()
hours_per_week = hours_scaler.fit_transform(X_train['hours-per-week'].values.reshape(-1,1))
hours_per_week = pd.DataFrame(hours_per_week, columns = ['hours-per-week'])
# ===========================================

# Task 5
# ===========================================
modeling_df = pd.concat([X_train.age, X_train['capital-gain'], X_train['capital-loss'], 
                         hours_per_week, categoricals], axis = 1)
# ===========================================

modeling_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>x0_?</th>
      <th>x0_Federal-gov</th>
      <th>x0_Local-gov</th>
      <th>x0_Never-worked</th>
      <th>x0_Private</th>
      <th>x0_Self-emp-inc</th>
      <th>...</th>
      <th>x7_Portugal</th>
      <th>x7_Puerto-Rico</th>
      <th>x7_Scotland</th>
      <th>x7_South</th>
      <th>x7_Taiwan</th>
      <th>x7_Thailand</th>
      <th>x7_Trinadad&amp;Tobago</th>
      <th>x7_United-States</th>
      <th>x7_Vietnam</th>
      <th>x7_Yugoslavia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-2.877472</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.775310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.207100</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.775310</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.036419</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 106 columns</p>
</div>



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

class BinAge(TransformerMixin, BaseEstimator):
    
    def fit(self, X, y=None):
        return self
    
    @vectorize
    def _bin_data(x):
        return int(x >= 45 and x <= 64)
        
    def transform(self, X):
        return self._bin_data(X)
```

**Now repeat the process for a `BinCapital` Transformer!**


```python
class BinCapital(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    @vectorize
    def _bin_data(x):
        return int(x > 0)
        
    def transform(self, X):
        return self._bin_data(X)
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




    array([0.80655141, 0.80818926, 0.80727934, 0.81143065, 0.80414998])




```python
cross_val_score(rf_pipeline, X_train, y_train)
```




    array([0.83730664, 0.83239308, 0.82984531, 0.84200946, 0.83272661])




```python
rf_pipeline.fit(X_train, y_train)
train_preds = rf_pipeline.predict(X_train)
val_preds = rf_pipeline.predict(X_val)
print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
print(f'Validation Accuracy: {accuracy_score(y_val, val_preds)}')
```

    Training Accuracy: 0.8443926764459652
    Validation Accuracy: 0.840467350949989


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
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x7f8dd51e90d0>),
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




    0.8415363197117354



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



### Visualize the pipeline!


```python
from sklearn import set_config

set_config(display="diagram")
model
```




<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f1755c1c-4854-44be-9d8d-08f71d3601b5" type="checkbox" ><label class="sk-toggleable__label" for="f1755c1c-4854-44be-9d8d-08f71d3601b5">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('columntransformer',
                 ColumnTransformer(transformers=[('binage', BinAge(), ['age']),
                                                 ('bincapital-1', BinCapital(),
                                                  ['capital-gain']),
                                                 ('bincapital-2', BinCapital(),
                                                  ['capital-loss']),
                                                 ('onehotencoder',
                                                  OneHotEncoder(handle_unknown='ignore',
                                                                sparse=False),
                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7f8dbaada3d0>),
                                                 ('standardscaler',
                                                  StandardScaler(),
                                                  ['hours-per-week'])])),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=10))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="95a350bb-1587-4ceb-bcf7-02824aa67481" type="checkbox" ><label class="sk-toggleable__label" for="95a350bb-1587-4ceb-bcf7-02824aa67481">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[('binage', BinAge(), ['age']),
                                ('bincapital-1', BinCapital(),
                                 ['capital-gain']),
                                ('bincapital-2', BinCapital(),
                                 ['capital-loss']),
                                ('onehotencoder',
                                 OneHotEncoder(handle_unknown='ignore',
                                               sparse=False),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f8dbaada3d0>),
                                ('standardscaler', StandardScaler(),
                                 ['hours-per-week'])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="7399b78b-42c3-4c8e-96cf-0107029eaa90" type="checkbox" ><label class="sk-toggleable__label" for="7399b78b-42c3-4c8e-96cf-0107029eaa90">binage</label><div class="sk-toggleable__content"><pre>['age']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="026d689b-309e-4a59-86a4-2987c41a5ae2" type="checkbox" ><label class="sk-toggleable__label" for="026d689b-309e-4a59-86a4-2987c41a5ae2">BinAge</label><div class="sk-toggleable__content"><pre>BinAge()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b3560970-b24b-4a64-8758-7dd6d4ccdcf1" type="checkbox" ><label class="sk-toggleable__label" for="b3560970-b24b-4a64-8758-7dd6d4ccdcf1">bincapital-1</label><div class="sk-toggleable__content"><pre>['capital-gain']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="8bd7b3bb-dbec-4abd-b737-5eac6ac66f44" type="checkbox" ><label class="sk-toggleable__label" for="8bd7b3bb-dbec-4abd-b737-5eac6ac66f44">BinCapital</label><div class="sk-toggleable__content"><pre>BinCapital()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="751ddbca-fdbb-46c9-ab21-e637f7cf4106" type="checkbox" ><label class="sk-toggleable__label" for="751ddbca-fdbb-46c9-ab21-e637f7cf4106">bincapital-2</label><div class="sk-toggleable__content"><pre>['capital-loss']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c80a8ae5-03d1-402f-8e55-d5e39947b337" type="checkbox" ><label class="sk-toggleable__label" for="c80a8ae5-03d1-402f-8e55-d5e39947b337">BinCapital</label><div class="sk-toggleable__content"><pre>BinCapital()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0857a5fa-ac88-4fb0-9c7e-b582860cc5a6" type="checkbox" ><label class="sk-toggleable__label" for="0857a5fa-ac88-4fb0-9c7e-b582860cc5a6">onehotencoder</label><div class="sk-toggleable__content"><pre><sklearn.compose._column_transformer.make_column_selector object at 0x7f8dbaada3d0></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="7ff8207b-663b-459a-9574-e312b5bdb439" type="checkbox" ><label class="sk-toggleable__label" for="7ff8207b-663b-459a-9574-e312b5bdb439">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown='ignore', sparse=False)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="4d280c59-acb8-46ff-b248-1890ff0ab714" type="checkbox" ><label class="sk-toggleable__label" for="4d280c59-acb8-46ff-b248-1890ff0ab714">standardscaler</label><div class="sk-toggleable__content"><pre>['hours-per-week']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="eea1b9c7-6cb5-49d4-bc48-3a63a3b64fc5" type="checkbox" ><label class="sk-toggleable__label" for="eea1b9c7-6cb5-49d4-bc48-3a63a3b64fc5">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="275a1e4d-9b21-40cc-a40a-15d169d466d3" type="checkbox" ><label class="sk-toggleable__label" for="275a1e4d-9b21-40cc-a40a-15d169d466d3">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_depth=10)</pre></div></div></div></div></div></div></div>


