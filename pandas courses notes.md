**建立新的dataframe，读写文件。**

```python
import pandas as pd
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])
# [[]]是按行来建立dataframe，{[]}是按列来创建
fruit_sales = pd.DataFrame([[35, 21], [41, 34]], columns=['Apples', 'Bananas'],
                index=['2017 Sales', '2018 Sales'])
               
quantities = ['4 cups', '1 cup', '2 large', '1 can']
items = ['Flour', 'Milk', 'Eggs', 'Spam']
recipe = pd.Series(quantities, index=items, name='Dinner')

reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)

animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals.to_csv("cows_and_goats.csv")
```

**取用dataframe中的元素**

```python
# 取columns
desc = reviews.description
# or
desc = reviews["description"]
first_description = reviews.description.iloc[0]

# 取 row
first_row = reviews.iloc[0]

# 取前十行的特定column
first_descriptions = reviews.loc[:9,'description']

# 取特定的rows
indices = [1, 2, 3, 5, 8]
sample_reviews = reviews.loc[indices]

# 取特定的行和列
cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = reviews.loc[indices, cols]

# 取前n行和特定列，注意loc是算进去0:99的99的，iloc不算99，只到98.
cols = ['country', 'variety']
df = reviews.loc[:99, cols]
# or
cols_idx = [0, 11]
df = reviews.iloc[:100, cols_idx]

# 按条件取rows
italian_wines = reviews[reviews.country == 'Italy']
# 多条件取rows，注意条件之间尽量都打括号
top_oceania_wines = reviews.loc[
    (reviews.country.isin(['Australia', 'New Zealand']))
    & (reviews.points >= 95)
]

```

**对dataframe进行计算操作或者apply一个funtion**

```
# 求中位数之类的，直接在后面加这种aggregation的function
median_points = reviews.points.median()
# 求unique的country
countries = reviews.country.unique()
# 每一种class里有几个数值
reviews_per_country = reviews.country.value_counts()
# 中心化！ 重要！
centered_price = reviews.price - reviews.price.mean()
# 先求最大的index，再返回dataframe定位需要的数据
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']
# 简单的函数可以用lamda表示，这里的map是对列进行的。
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

# 复杂的函数可以写好了之后再apply，可以对行进行，也可以对列进行，更改参数axis就好。
# 0 or ‘index’: apply function to each column.
# 1 or ‘columns’: apply function to each row.

def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')
```

**grouping and sorting**

```python
reviews_written = reviews.groupby('taster_twitter_handle').size()
# or
reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()

price_extremes = reviews.groupby('variety').price.agg([min, max])

sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()

country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)






```

**Missing Values**

```python
# 取得数据列的type
dtype = reviews.points.dtype

# 变换数据列的type
point_strings = reviews.points.astype(str)

# 取得数据列null的数量
missing_price_reviews = reviews[reviews.price.isnull()]
n_missing_prices = len(missing_price_reviews)
# Cute alternative solution: if we sum a boolean series, True is treated as 1 and False as 0
n_missing_prices = reviews.price.isnull().sum()
# or equivalently:
n_missing_prices = pd.isnull(reviews.price).sum()

# 改变null的值，然后sort
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)

#在这一列里改class的名字
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
```

**Combining and Renaming**

```python
# 对column的名字重命名，可以是一个字典object，或者如下创造字典
renamed = reviews.rename(columns=dict(region_1='region', region_2='locale'))

# 这个是给column和row加名字
reindexed = reviews.rename_axis('wines', axis='rows')

# 给index改名，不太常用
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# 结合两个dataframe
combined_products = pd.concat([gaming_products, movie_products])

# 使用某个共同的index来进行结合
powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))

# 可以加上suffix，此为选项，具体参考document
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')
```

总的来说，知道这些基本功能之后，在具体应用中还是需要去查阅doc。

