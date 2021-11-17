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


```python
cross_val_score(rf_pipeline, X_train, y_train)
```


```python
rf_pipeline.fit(X_train, y_train)
train_preds = rf_pipeline.predict(X_train)
val_preds = rf_pipeline.predict(X_val)
print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
print(f'Validation Accuracy: {accuracy_score(y_val, val_preds)}')
```

Finally, we can fit the final pipeline on all of the data and test it on an additional hold out set!


```python
rf_pipeline.fit(df.drop('income', axis = 1), df.income)
```

Load in the hold out set and make predictions!


```python
# Import holdout data
test = pd.read_csv('data/test.csv')
# Seperate features from the target
X_test, y_test = test.drop(columns=['income']), test.income
# Score the model
rf_pipeline.score(X_test, y_test)
```

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
