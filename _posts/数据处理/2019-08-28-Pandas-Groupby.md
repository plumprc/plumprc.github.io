---
title: Pandas-groupby
date: 2019-08-28 16:22:54
categories:
- 数据处理
tags:
- 数据处理
---

# Pandas-groupby


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('ggplot')
path = 'https://raw.githubusercontent.com/HoijanLai/dataset/master/PoliceKillingsUS.csv'
data = pd.read_csv(path)
data.sample(3) # randomly
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
      <th>name</th>
      <th>date</th>
      <th>race</th>
      <th>age</th>
      <th>signs_of_mental_illness</th>
      <th>flee</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1613</th>
      <td>Marcelo Luna</td>
      <td>19/08/16</td>
      <td>H</td>
      <td>47.0</td>
      <td>True</td>
      <td>Not fleeing</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Thaddeus McCarroll</td>
      <td>17/04/15</td>
      <td>B</td>
      <td>23.0</td>
      <td>True</td>
      <td>Not fleeing</td>
    </tr>
    <tr>
      <th>1379</th>
      <td>Doll Pierre-Louis</td>
      <td>25/05/16</td>
      <td>B</td>
      <td>24.0</td>
      <td>False</td>
      <td>Car</td>
    </tr>
  </tbody>
</table>
</div>



---
## datas = data.groupby('A')
![datas = data.groupby('A')](https://upload-images.jianshu.io/upload_images/2862169-51af7d4ae64c2f78.png?imageMogr2/auto-orient/)


```python
datas = data.groupby('race')
print(type(datas))
```

    <class 'pandas.core.groupby.generic.DataFrameGroupBy'>
    


```python
data.groupby('race')['age'].mean()
```




    race
    A    36.605263
    B    31.635468
    H    32.995157
    N    30.451613
    O    33.071429
    W    40.046980
    Name: age, dtype: float64




```python
data.groupby('race')['signs_of_mental_illness'].value_counts()
```




    race  signs_of_mental_illness
    A     False                       29
          True                        10
    B     False                      523
          True                        95
    H     False                      338
          True                        85
    N     False                       23
          True                         8
    O     False                       21
          True                         7
    W     False                      819
          True                       382
    Name: signs_of_mental_illness, dtype: int64




```python
data.groupby('race')['signs_of_mental_illness'].value_counts().unstack()
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
      <th>signs_of_mental_illness</th>
      <th>False</th>
      <th>True</th>
    </tr>
    <tr>
      <th>race</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>29</td>
      <td>10</td>
    </tr>
    <tr>
      <th>B</th>
      <td>523</td>
      <td>95</td>
    </tr>
    <tr>
      <th>H</th>
      <td>338</td>
      <td>85</td>
    </tr>
    <tr>
      <th>N</th>
      <td>23</td>
      <td>8</td>
    </tr>
    <tr>
      <th>O</th>
      <td>21</td>
      <td>7</td>
    </tr>
    <tr>
      <th>W</th>
      <td>819</td>
      <td>382</td>
    </tr>
  </tbody>
</table>
</div>



---
## Visualization
### Scene1 : The discrete distribution about flee method in different races


```python
# Tranditional way
# iterator -> filter -> plot each of them
races = np.sort(data['race'].dropna().unique())
fig, axes = plt.subplots(1, len(races), figsize=(24, 4), sharey=True)
for ax, race in zip(axes, races):
    data[data['race']==race]['flee'].value_counts().sort_index().plot(kind='bar', ax=ax, title=race)
```


![png](https://raw.githubusercontent.com/plumprc/plumprc.github.io/master/_posts/%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/Pandas-Groupby_files/Pandas-Groupby_8_0.png)



```python
# Amazing Groupby!
data.groupby('race')['flee'].value_counts().unstack().plot(kind='bar', figsize=(20, 4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x275e4d96b00>




![png](https://raw.githubusercontent.com/plumprc/plumprc.github.io/master/_posts/%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/Pandas-Groupby_files/Pandas-Groupby_9_1.png)


---
### Scene2 : The continuous distribution about flee method in different ages


```python
data.groupby('flee')['age'].plot(kind='kde', legend=True, figsize=(20, 5))
```




    flee
    Car            AxesSubplot(0.125,0.125;0.775x0.755)
    Foot           AxesSubplot(0.125,0.125;0.775x0.755)
    Not fleeing    AxesSubplot(0.125,0.125;0.775x0.755)
    Other          AxesSubplot(0.125,0.125;0.775x0.755)
    Name: age, dtype: object




![png](https://raw.githubusercontent.com/plumprc/plumprc.github.io/master/_posts/%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/Pandas-Groupby_files/Pandas-Groupby_11_1.png)


---
### Scene3 : Apply different operations on different columns


```python
# aggragate
data.groupby('race').agg({'age': np.median, 'signs_of_mental_illness': np.mean})
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
      <th>signs_of_mental_illness</th>
    </tr>
    <tr>
      <th>race</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>35.0</td>
      <td>0.256410</td>
    </tr>
    <tr>
      <th>B</th>
      <td>30.0</td>
      <td>0.153722</td>
    </tr>
    <tr>
      <th>H</th>
      <td>31.0</td>
      <td>0.200946</td>
    </tr>
    <tr>
      <th>N</th>
      <td>29.0</td>
      <td>0.258065</td>
    </tr>
    <tr>
      <th>O</th>
      <td>29.5</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>W</th>
      <td>38.0</td>
      <td>0.318068</td>
    </tr>
  </tbody>
</table>
</div>



---
### Scene4 : Apply different operations on one column


```python
data.groupby('flee')['age'].agg([np.mean, np.median, np.std])
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
      <th>mean</th>
      <th>median</th>
      <th>std</th>
    </tr>
    <tr>
      <th>flee</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Car</th>
      <td>33.911765</td>
      <td>33.0</td>
      <td>11.174234</td>
    </tr>
    <tr>
      <th>Foot</th>
      <td>30.972222</td>
      <td>30.0</td>
      <td>10.193900</td>
    </tr>
    <tr>
      <th>Not fleeing</th>
      <td>38.334753</td>
      <td>36.0</td>
      <td>13.527702</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>33.239130</td>
      <td>33.0</td>
      <td>9.932043</td>
    </tr>
  </tbody>
</table>
</div>
