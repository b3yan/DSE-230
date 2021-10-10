#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
pathToFile = os.environ['DATA_FILE_PATH']


# ### 1. (From JupyterLab terminal) Copy data file(BookReviews_1M.txt) from local FS to HDFS

# In[1]:


get_ipython().system('pwd')


# In[2]:


get_ipython().system('ls')


# In[4]:


get_ipython().system('hadoop fs -mkdir /hdfs_folder')


# In[5]:


get_ipython().system('hadoop fs -copyFromLocal BookReviews_1M.txt /hdfs_folder')


# ### 2. Start spark session

# In[6]:


import pyspark
from pyspark.sql import SparkSession

conf = pyspark.SparkConf().setAll([
         ('spark.master', 'local[1]'),
         ('spark.app.name', 'App Name')])
    
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# ### 3. Read data from HDFS into Spark DataFrame

# In[7]:


df = spark.read.text('/hdfs_folder/BookReviews_1M.txt')


# ### 4. Print number of lines read in

# In[8]:


df_count = df.count()
print('Number of lines read in:\n', df_count)


# ### 5. Show first 20 lines using pyspark.sql.DataFrame.show

# In[9]:


df.show(n = 20, truncate = False)


# ### 6. Stop spark session

# In[10]:


spark.stop()


# In[ ]:




