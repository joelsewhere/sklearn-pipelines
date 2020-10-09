# Creating a Custom Sklearn Pipeline
> How to include custom data preprocessing steps in an sklearn pipeline. 

In this notebook we will import the [income classification dataset](https://www.kaggle.com/lodetomasi1995/income-classification/data), review common preprocessing steps, and then introduce how those steps can be included in an sklearn pipeline. 

![](https://media.giphy.com/media/Jwp4sxM0Rjk1W/giphy.gif)


```python
# Standard Imports
import pandas as pd
import pickle

# Transformers
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Modeling Evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Pipelines
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline

# Machine Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```


```python
df = pd.read_csv('data/income.csv')
df.head()
```

# Breakout Rooms


### Group 1

Write a function called `bin_middle_age` that can be applied to the `age` column and returns a 1 if the age is 45-64 and a zero for every other age. 

### Group 2

Write a function called `bin_capital` that can be applied to the `capital_gain` and `capital_loss` columns and returns a 1 if the input is more than zero and a 0 for anything else.

### Group 3

Please write code to fit a one hot encoder to all of the object datatypes. Transform the object columns and turn them into a dataframe. For this final step, I'll give you two clues: "sparse" and "dense". Only one of them will be needed.

### Group 4

Please write code to scale the `'hours_per_week'` column. Because you are scaling, please write code that does not lead to data leakage.


```python
# Group 1
def bin_middle_age(age):
    pass

df['age'] = df.age.apply(bin_middle_age)

# Group 2
def bin_capital(x):
    pass

df['capital_gain'] = df.capital_gain.apply(bin_capital)
df['capital_loss'] = df.capital_loss.apply(bin_capital)

X_train, X_test, y_train, y_test = train_test_split(df.drop('income', axis = 1), 
                                                    df.income,
                                                    random_state = 2020)  
X_train.reset_index(drop=True, inplace=True)

# Group 3
    pass

# Group 4
    pass


modeling_df = pd.concat([X_train.age, X_train.capital_gain, X_train.capital_loss, 
                         hours_per_week, categoricals], axis = 1)

modeling_df.head()
```

# Move all of this into a Pipeline

Above we used two sklearn transformers and two custom functions to format our dataframe. This means, that we will need to create two custom transformers. The sklearn transformers can be used as they are.

To do this, we will create a class called `BinAge` that inherits from the sklearn classes, `TransformerMixin` and `BaseEstimator`. This class should have the following methods:
1. `__init__`
    - This method only needs to exist. No code needs to be added to the method.
2. `fit`
    - This method should have three arguments
        1. self
        2. `X`.
        3. `y=None`
    - This method should return `self`.
3. `_bin_data`
    - This method is our function for binning the age column
4. `_to_df`
    - This is a helper function to transform the data to a dataframe.
    - This method should check if the input is a dataframe and return a dataframe
5. `transform`
    - This method should have two arguments
        1. self
        2. `X`
    - This method should turn X to a dataframe. 
    - This method should apply the `_bin_data` method
    - Return the binned data


```python
class BinAge():
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
                                      (BinCapital(), ['capital_gain']),
                                      (BinCapital(), ['capital_loss']),
                                      (OneHotEncoder(),
                                       make_column_selector(dtype_include=object)),
                                      (StandardScaler(), ['hours_per_week']),
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
test_preds = rf_pipeline.predict(X_test)
print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
print(f'Testing Accuracy: {accuracy_score(y_test, test_preds)}')
```

Finally, we can fit the final pipeline on all of the data and test it on an additional hold out set!


```python
rf_pipeline.fit(df.drop('income', axis = 1), df.income)
```

Load in the hold out set and make predictions!


```python
validation = pd.read_csv('data/validation_features.csv')
val_preds = rf_pipeline.predict(validation)
y_val = pd.read_csv('data/validation_target.csv').iloc[:,0]
accuracy_score(y_val, val_preds)
```
