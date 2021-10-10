#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Load-Libraries" data-toc-modified-id="Load-Libraries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load Libraries</a></span></li><li><span><a href="#Initialize-pyspark-framework" data-toc-modified-id="Initialize-pyspark-framework-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Initialize pyspark framework</a></span></li><li><span><a href="#Load-data" data-toc-modified-id="Load-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Overview-of-Dataset" data-toc-modified-id="Overview-of-Dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Overview of Dataset</a></span><ul class="toc-item"><li><span><a href="#Data-schema" data-toc-modified-id="Data-schema-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Data schema</a></span></li><li><span><a href="#Columns-overview" data-toc-modified-id="Columns-overview-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Columns overview</a></span></li><li><span><a href="#Summary-statistics-for-numeric-variables" data-toc-modified-id="Summary-statistics-for-numeric-variables-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Summary statistics for numeric variables</a></span></li><li><span><a href="#Show-data-and-data-count" data-toc-modified-id="Show-data-and-data-count-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Show data and data count</a></span></li></ul></li><li><span><a href="#Correlations" data-toc-modified-id="Correlations-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Correlations</a></span><ul class="toc-item"><li><span><a href="#Checking-Correlations-between-independent-variables" data-toc-modified-id="Checking-Correlations-between-independent-variables-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Checking Correlations between independent variables</a></span></li><li><span><a href="#Explore-relationships-across-the-entire-dataset" data-toc-modified-id="Explore-relationships-across-the-entire-dataset-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Explore relationships across the entire dataset</a></span></li></ul></li><li><span><a href="#Distribution-of-Data" data-toc-modified-id="Distribution-of-Data-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Distribution of Data</a></span></li><li><span><a href="#Common-trend" data-toc-modified-id="Common-trend-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Common trend</a></span></li><li><span><a href="#Heatmap-for-comprehensive-overview" data-toc-modified-id="Heatmap-for-comprehensive-overview-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Heatmap for comprehensive overview</a></span></li><li><span><a href="#Stop-the-spark-session" data-toc-modified-id="Stop-the-spark-session-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Stop the spark session</a></span></li></ul></div>

# # Load Libraries

# In[1]:


# Import PySpark related modules
import pyspark
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions
from pyspark.sql.functions import lit, desc, col, size, array_contains, isnan, udf, hour, array_min, array_max, countDistinct, quarter
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

# Import other modules not related to PySpark
import os
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import math
from IPython.core.interactiveshell import InteractiveShell
from datetime import *
import seaborn as sns
import statistics as stats
# This helps auto print out the items without explixitly using 'print'
InteractiveShell.ast_node_interactivity = "all" 
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import figure
import numpy as np
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings("ignore")


# # Initialize pyspark framework

# In[2]:


conf = pyspark.SparkConf().setAll([('spark.master', 'local[*]'),
                                   ('spark.app.name', 'Python Spark SQL Demo')])
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# # Load data

# In[3]:


get_ipython().system('pwd')


# In[4]:


get_ipython().system('ls')


# In[5]:


get_ipython().system('hadoop fs -mkdir /data')


# In[6]:


get_ipython().system('hadoop fs -copyFromLocal products_dataset.csv /data')


# In[7]:


get_ipython().system('hadoop fs -copyFromLocal product_category_name_translation.csv /data')


# In[8]:


get_ipython().system('hadoop fs -copyFromLocal customers_dataset.csv /data')


# In[9]:


get_ipython().system('hadoop fs -copyFromLocal sellers_dataset.csv /data')


# In[10]:


get_ipython().system('hadoop fs -copyFromLocal orders_dataset.csv /data')


# In[11]:


get_ipython().system('hadoop fs -copyFromLocal order_payments_dataset.csv /data')


# In[12]:


get_ipython().system('hadoop fs -copyFromLocal order_items_dataset.csv /data')


# In[13]:


get_ipython().system('hadoop fs -copyFromLocal geolocation_dataset.csv /data')


# In[14]:


get_ipython().system('hadoop fs -copyFromLocal customer_reviews_dataset.csv /data')


# In[15]:


DATA_PATH="hdfs:///data/"
products_dataset = spark.read.csv(DATA_PATH+"products_dataset.csv", header=True, inferSchema = True)
product_category_name_translation = spark.read.csv(DATA_PATH+"product_category_name_translation.csv", header=True, inferSchema = True)
customers_dataset = spark.read.csv(DATA_PATH+"customers_dataset.csv", header=True, inferSchema = True)
sellers_dataset = spark.read.csv(DATA_PATH+"sellers_dataset.csv", header=True, inferSchema = True)
orders_dataset = spark.read.csv(DATA_PATH+"orders_dataset.csv", header=True, inferSchema = True)
order_payments_dataset = spark.read.csv(DATA_PATH+"order_payments_dataset.csv", header=True, inferSchema = True)
order_items_dataset = spark.read.csv(DATA_PATH+"order_items_dataset.csv", header=True, inferSchema = True)
geolocation_dataset = spark.read.csv(DATA_PATH+"geolocation_dataset.csv", header=True, inferSchema = True)
customer_reviews_dataset = spark.read.csv(DATA_PATH+"customer_reviews_dataset.csv", header=True, inferSchema = True)


# # Overview of Dataset

# ## Data schema

# In[16]:


print('Data overview')
products_dataset.printSchema()


# In[17]:


print('Data overview')
product_category_name_translation.printSchema()


# In[18]:


print('Data overview')
customers_dataset.printSchema()


# In[19]:


print('Data overview')
sellers_dataset.printSchema()


# In[20]:


print('Data overview')
orders_dataset.printSchema()


# In[21]:


print('Data overview')
order_payments_dataset.printSchema()


# In[22]:


print('Data overview')
order_items_dataset.printSchema()


# In[23]:


print('Data overview')
geolocation_dataset.printSchema()


# In[24]:


print('Data overview')
customer_reviews_dataset.printSchema()


# ## Columns overview

# In[25]:


print('Columns overview')
pd.DataFrame(products_dataset.dtypes, columns = ['Column Name','Data type'])


# In[26]:


print('Columns overview')
pd.DataFrame(product_category_name_translation.dtypes, columns = ['Column Name','Data type'])


# In[27]:


print('Columns overview')
pd.DataFrame(customers_dataset.dtypes, columns = ['Column Name','Data type'])


# In[28]:


print('Columns overview')
pd.DataFrame(sellers_dataset.dtypes, columns = ['Column Name','Data type'])


# In[29]:


print('Columns overview')
pd.DataFrame(orders_dataset.dtypes, columns = ['Column Name','Data type'])


# In[30]:


print('Columns overview')
pd.DataFrame(order_payments_dataset.dtypes, columns = ['Column Name','Data type'])


# In[31]:


print('Columns overview')
pd.DataFrame(order_items_dataset.dtypes, columns = ['Column Name','Data type'])


# In[32]:


print('Columns overview')
pd.DataFrame(geolocation_dataset.dtypes, columns = ['Column Name','Data type'])


# In[33]:


print('Columns overview')
pd.DataFrame(customer_reviews_dataset.dtypes, columns = ['Column Name','Data type'])


# ## Summary statistics for numeric variables

# In[34]:


print('Data frame describe (string and numeric columns only):')
products_dataset.describe().toPandas()


# In[35]:


print('Data frame describe (string and numeric columns only):')
product_category_name_translation.describe().toPandas()


# In[36]:


print('Data frame describe (string and numeric columns only):')
customers_dataset.describe().toPandas()


# In[37]:


print('Data frame describe (string and numeric columns only):')
sellers_dataset.describe().toPandas()


# In[38]:


print('Data frame describe (string and numeric columns only):')
orders_dataset.describe().toPandas()


# In[39]:


print('Data frame describe (string and numeric columns only):')
order_payments_dataset.describe().toPandas()


# In[40]:


print('Data frame describe (string and numeric columns only):')
order_items_dataset.describe().toPandas()


# In[41]:


print('Data frame describe (string and numeric columns only):')
geolocation_dataset.describe().toPandas()


# In[42]:


print('Data frame describe (string and numeric columns only):')
customer_reviews_dataset.describe().toPandas()


# ## Show data and data count

# In[43]:


print(f'There are total {products_dataset.count()} row, Let print first 2 data rows:\n')
products_dataset.limit(2).toPandas()


# In[44]:


print(f'There are total {product_category_name_translation.count()} row, Let print first 2 data rows:\n')
product_category_name_translation.limit(2).toPandas()


# In[45]:


print(f'There are total {customers_dataset.count()} row, Let print first 2 data rows:\n')
customers_dataset.limit(2).toPandas()


# In[46]:


print(f'There are total {sellers_dataset.count()} row, Let print first 2 data rows:\n')
sellers_dataset.limit(2).toPandas()


# In[47]:


print(f'There are total {orders_dataset.count()} row, Let print first 2 data rows:\n')
orders_dataset.limit(2).toPandas()


# In[48]:


print(f'There are total {order_payments_dataset.count()} row, Let print first 2 data rows:\n')
order_payments_dataset.limit(2).toPandas()


# In[49]:


print(f'There are total {order_items_dataset.count()} row, Let print first 2 data rows:\n')
order_items_dataset.limit(2).toPandas()


# In[50]:


print(f'There are total {geolocation_dataset.count()} row, Let print first 2 data rows:\n')
geolocation_dataset.limit(2).toPandas()


# In[51]:


print(f'There are total {customer_reviews_dataset.count()} row, Let print first 2 data rows:\n')
customer_reviews_dataset.limit(2).toPandas()


# # Correlations

# ## Checking Correlations between independent variables

# In[52]:


# Merge these two dataframes together: products_dataset, product_category_name_translation.
df_merge_product_and_category = products_dataset.join(product_category_name_translation, on=['product_category_name'], how='inner')
df_merge_product_and_category = df_merge_product_and_category.drop('product_category_name')
df_merge_product_and_category = df_merge_product_and_category.drop_duplicates(['product_id'])
df_merge_product_and_category = df_merge_product_and_category.dropna()

df_merge_product_and_category.show(2)


# In[53]:


# Checking Correlations between independent variables
numeric_features = [t[0] for t in df_merge_product_and_category.dtypes if t[1] == 'int']
numeric_data = df_merge_product_and_category.select(numeric_features).toPandas()

axs = scatter_matrix(numeric_data, figsize=(25, 25));

# Rotate axis labels and remove axis ticks
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)


# In[54]:


# Merge these three dataframes together: orders_dataset, order_payments_dataset, order_items_dataset.
df_merge_1 = orders_dataset.join(order_payments_dataset, on=['order_id'], how='inner')
df_merge_2 = df_merge_1.join(order_items_dataset, on=['order_id'], how='inner')

#df_merge = df_merge_2.drop('order_approved_at', 'order_carrier_delivery_date', 'order_customer_delivery_date', 'order_estimated_delivery_date', 'payment_sequential', 'payment_type', 'payment_installments','shipping_limit_date')
df_merge = df_merge_2.drop_duplicates(['order_id'])
df_merge_order = df_merge.dropna()
df_merge_order = df_merge.select('order_id', 'order_item_id', 'customer_id', 'seller_id', 'product_id', 'order_status', 'order_purchase_timestamp', 'price', 'freight_value', 'payment_value')

df_merge_order.show(2)


# In[55]:


# Checking Correlations between independent variables
numeric_features = [t[0] for t in df_merge_order.dtypes if t[1] == 'double']
numeric_data = df_merge_order.select(numeric_features).toPandas()

axs = scatter_matrix(numeric_data, figsize=(25, 25));

# Rotate axis labels and remove axis ticks
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)


# ## Explore relationships across the entire dataset

# In[56]:


# merge all above dataframes together
merge_df = df_merge_product_and_category.join(df_merge_order, on=["product_id"], how='inner')
merge_df = merge_df.select('product_id', 'price', 'freight_value', 'payment_value','product_name_lenght','product_description_lenght', 'product_photos_qty', 'product_weight_g',                          'product_length_cm', 'product_height_cm', 'product_width_cm','product_category_name_english','order_purchase_timestamp')

merge_df.show(2)


# In[57]:


sns.pairplot(merge_df.toPandas())
plt.show()


# # Distribution of Data

# In[58]:


plot = df_merge_product_and_category.toPandas().boxplot(figsize = (25,10))


# In[59]:


df_merge_order.toPandas().boxplot(figsize = (25,10))


# In[60]:


plot = merge_df.toPandas().boxplot(column='payment_value', by='product_photos_qty', fontsize='small', figsize=(25,10))


# The boxplot showed that except the outliers, product photos quantity within 10 have higner payment values.

# In[61]:


df_merge_new = merge_df.select('product_id', 'order_purchase_timestamp', 'price', 'freight_value','payment_value').orderBy('seller_id', 'product_id')
date_col = df_merge_new.select(date_format(col('order_purchase_timestamp'),"yyyy-MM-dd").alias('Date').cast("date"))
date_col = date_col.withColumn("id", monotonically_increasing_id())
df_merge_new = df_merge_new.withColumn("id", monotonically_increasing_id())
df3 = df_merge_new.join(date_col, on=["id"], how="left").drop("id", "order_purchase_timestamp")
df3 = df3.dropna()
df3.show(2)


# In[62]:


df = df3.toPandas()
df[['Date','price', 'freight_value','payment_value']].plot(x='Date', subplots=True, figsize=(28,28))
plt.show()


# # Common trend

# In[63]:


merge_df.toPandas().hist(figsize = (25,15), bins = 80)


# # Heatmap for comprehensive overview

# In[64]:


Var_Corr = merge_df.toPandas().corr()
# plot the heatmap and annotation on it
plt.figure(figsize = (25,25))
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)


# # Stop the spark session

# In[65]:


spark.stop()


# In[ ]:




