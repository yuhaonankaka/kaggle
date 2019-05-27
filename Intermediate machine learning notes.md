- 导入必要的module

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learntools.core import *
```

- 数据路径

```python
iowa_file_path = '../input/train.csv'
```

- 导入数据

```python
home_data = pd.read_csv(iowa_file_path)
```

- 创建target，这个是要预测的结果column

```python
y = home_data.SalePrice
```

- 预测需要的columns，作为信息的输入

```python
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
```

- 将数据分为validation和training

```python
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2,random_state=1)
```

- preprocessing

```python
# 可以drop一些column或者row，axis = 1 是column， 0 是row
melb_predictors = data.drop(['Price'], axis=1)
# 可以选择dtype，object一般是text
X = melb_predictors.select_dtypes(exclude=['object'])
# 可以dorp row，一般是drop掉那些没有target value的row
# Drop rows with any empty cells
my_dataframe.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
```

- Missing values

```python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
                     # 这里是说这个col的数据是否为null，返回一个array of boolean，然后any()来检测是否有任何一个true在这个array里。

# 1. Drop Columns with Missing Values
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
# 2. Imputation 只是把没有的数据用mean或者什么别的补上。
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
# 不能用valid的data来fit transform，不然会data leakage
# 这里很重要！Imputation会去掉column names！要重新加上！
# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# 3. An Extension to Imputation 这个可以创造新的column，表示之前有没有missing，也会补上一个value
# Make new columns indicating what will be imputed
for col in cols_with_missing: # 创造了新的列
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
# 接下来和第二种方法一样，把missing的value补上。记得加上column names！
```

- Categorical variable

```python
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# 1. drop
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# 2. Label Encoding 不会产生新的column，把catagory变成0，1，2，3，4...
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

# 3. One-Hot Encoding 会产生新的column，把每种class变成一个column，用0，1表示。
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# handle_unknown是说如果validation里有一些class在training data里没有。这里是说class，也就是说在这个column中有一些值，val data中有，但training data是没有的。后面有另一个，就是说有些columns直接training data里就没有，那就要用后面的方法，把这个column去掉。
# We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
# sparse是返回一个numpy array而不是一个sparse matrix
# 注：这里是用所有catagorical的columns创造了一个新的matrix，之后还要和numerical的matrix合并，这里是X_train[object_cols]
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
# 重要！ one hot encoding会去除index，要重新加上！
# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# 有时候有一些column，validation data里有，但training data没有
# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

```



- 训练模型

```python
# 建立模型先
rf_model = RandomForestRegressor(random_state=1)
# fit一下，让模型成为适合这一波数据的模型
rf_model.fit(train_X, train_y)
# 预测一波
rf_val_predictions = rf_model.predict(val_X)
# 用mean_absolute_error来看看我们的预测和结果的差距，有很多方法来检查好坏。
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
```

- 真正用在预测新的数据时
- 注：现在可以直接用选好的参数，使用全部的数据创建一个新的model，使用所有数据集进行训练，因为之前是分为了training和validation，决定了参数和preprocessing的方式之后，可以使用所有数据集进行新的完整的训练。

```python
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X,y)
```

- 用完整dataset训练出的模型来预测想要测试的数据

```python
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)
# 写入文件中
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```

- tricks

```python
.copy() 可以把数据复制一下，就不用修改原来的数据了
# Shape of training data (num_rows, num_columns)
print(X_train.shape)
# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
# 可以inplace，就直接改变原始数据了
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
# 查看一些数据
X_train.head()
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
```

- PipeLine

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# 直接用pipeline来fit
# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```

- Cross Validation

```python
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
# scoring还有很多别的选项
# 用pipeline代入进去，pipeline是best practice
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("Average MAE score (across experiments):")
print(scores.mean())
```

- **XGBoost** （**extreme gradient boosting**）

```python
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
# n_estimators代表最多个模型，learning_rate每次的步长，步长小，可以多一点n estimators，
# learning rate：we can multiply the predictions from each model by a small number
n jobs是并行任务数量

my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
# early_stopping_rounds代表这么多次以后还不增加准确度，就停止迭代了。
# eval_set表示迭代的检查数据集
             
from sklearn.metrics import mean_absolute_error
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
```

- Data Leakage

  这是在说，validation结果看似好但其实真实的预测结果一点也不好。

  - Target leakage

  这是在说，有些作为输入的信息，其实是用了target的信息之后取得的。

  - Train-Test Contamination

  这是在说，例如training的时候的processing，不能把validation的data来一起fit transform，否则结果就很好，但是其实不好

总的来说，我们把leakage的column都drop掉就好

然后就是fit transform的时候不要把validation和training data一起fit transform

training data 用fit transform

validation data 用transform（因为之前已经训练好了这个transform的object）

```python
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \%((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \%(( expenditures_cardholders == 0).mean()))
```

