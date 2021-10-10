#!/usr/bin/env python
# coding: utf-8

# # DSE 230: Programming Assignment 4.1 - K-Means Cluster Analysis
# ---
# #### Tasks:
# - Work with `minute_weather.csv`
#     - Use scikit-learn to perform k-means clustering (25%)
#     - Explore parallelism with scikit-learn for k-means clustering (10%)
#     - Explore parallelism with dask for k-means clustering (65%)
# - Submission on Gradescope (2 files)
#   - Completed notebook (.ipynb) or PDF with results under **PA4.1 Notebook**
#     - Make sure that all expected outputs are present
#   - An executable script (.py) exported from this notebook under **PA4.1**
# 
# #### Due date: Friday 5/28/2021 at 11:59 PM PST

# ## Setup

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 
# ## Scikit-Learn (25%)
# ---

# **1.1** (5%) Load Data
# - Load the "minute_weather.csv" into the Pandas dataframe
# - Drop the two columns ["rowID", "hpwren_timestamp"] from the dataframe
# - Print out the column names (features) from the output of the previous step

# In[2]:


# Load the "minute_weather.csv" into the Pandas dataframe
df = pd.read_csv("minute_weather.csv")


# In[3]:


# Drop the two columns ["rowID", "hpwren_timestamp"] from the dataframe
df = df.drop(["rowID", "hpwren_timestamp"], axis = 1)


# In[4]:


# Print out the column names (features) from the output of the previous step
print(df.columns)


# In[5]:


# Drop null values
df = df.dropna()


# **1.2** (5%) Data preprocessing and normalization using sklearn
# - Perform train and test split with 80% of the original dataset being the training dataset and 20% of the original dataset being the testing dataset.
#     * Pass `random_state=seed` to `train_test_split` for reproducing results
# - Print the number of samples from both train and test dataset, and the summary statistics of training dataset.
# - Perform feature normalization on both the train dataset and the test dataset using StandardScaler from sklearn library. Only **train** data should be used for scaling
# - Print out the mean and standard deviation along the feature columns of both the train and the test dataset.
# 
# (your output of the mean and std should be a vector of shape (1, number of features) make sure you clearly label your results)

# In[6]:


seed=30
# Perform train and test split with 80% of the original dataset being the training dataset and 20% of the original dataset being the testing dataset.

train, test = train_test_split(df, test_size=0.2, random_state=seed)


# In[7]:


# Print the number of samples from both train and test dataset, and the summary statistics of training dataset.
print("Number of train samples:", len(train))
print("Number of test samples:", len(test))

print("\nSummary statistics of training dataset:\n", train.describe())


# In[8]:


# Perform feature normalization on both the train dataset and the test dataset using StandardScaler from sklearn library. Only train data should be used for scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scalerModel = scaler.fit(train)

train_df = scalerModel.transform(train)
test_df = scalerModel.transform(test)


# In[9]:


# Print out the mean and standard deviation along the feature columns of both the train and the test dataset.
# your output of the mean and std should be a vector of shape (1, number of features) make sure you clearly label your results

train_mean = pd.DataFrame(train.mean()).transpose()
print("Mean of the train dataset:\n")
train_mean


# In[10]:


test_mean = pd.DataFrame(test.mean()).transpose()
print("Mean of the test dataset:\n")
test_mean


# In[11]:


train_std = pd.DataFrame(train.std()).transpose()
print("Standard deviation of the train dataset:\n")
train_std


# In[12]:


test_std = pd.DataFrame(test.std()).transpose()
print("Standard deviation of the test dataset:\n")
test_std


# ### Build Clustering Model
# **1.4** (10%) KMeans Clustering model with sklearn
# - Use the normalized training dataset to fit a K-means model with 9 clusters
#     * Pass `random_state=seed` to `KMeans` for reproducing results
# - Print out the cluster centers found by the model
# - Print out the computational performance by adding "%%time" at the top of the cell

# In[13]:


get_ipython().run_cell_magic('time', '', "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=9, random_state=seed).fit(train_df)\nprint('cluster centers:\\n', kmeans.cluster_centers_)")


# ### Evaluate Model
# 
# **1.5** (5%) Evaluate KMeans clustering model with sklearn
# - Print out the inertia_ variable of the model, and explain what the number means in KMeans model
# - Print out the within-cluster sum of squares (WSSE) on the train and test
# 
# Check documentations on KMeans at https://scikit-learn.org/stable/modules/clustering.html

# In[14]:


print('inertia variable =', kmeans.inertia_, '\nIt means how far away the points within a cluster are. Lower values are better and zero is optimal.')


# In[15]:


print('WSSE_train =', -kmeans.score(train_df))
print('WSSE_test =', -kmeans.score(test_df))


# ---
# ## Parallelism with Scikit-Learn (10%)
# **2.1** (10%) Single machine parallelism using **all** the cores
# - Fit the model with single-machine parallelism using scikit-learn and joblib (via `n_jobs` parameter)
#     * Pass `random_state=seed` to `KMeans` for reproducing results
# - Print out the WSSE on train and test
# - Use %%time to print out the computational performance
# 
# Note that your model's parameters and seed setting should remain the same from the previous questions

# In[16]:


get_ipython().run_cell_magic('time', '', "from sklearn.cluster import KMeans\nimport joblib\nkmeans_p = KMeans(n_clusters = 9, random_state = seed)\n\nwith joblib.Parallel(n_jobs=-1):\n    kmeans_p.fit(train_df)\n\nprint('WSSE_train =', -kmeans_p.score(train_df))\nprint('WSSE_test =', -kmeans_p.score(test_df))")


# ---
# ## Parallelism with Dask (65%)
# Multi-machine parallelism using Dask's scalable k-means algorithm

# ### Create and connect to client
# **3.1** (5%) Setup the Dask distributed client
# - Create a Dask distributed client with 2 workers
# - Print out the Dask client information

# In[17]:


import joblib
from dask.distributed import Client

# Start and connect to local client
client = Client(n_workers=2)

# client = Client("scheduler-address:8786")  # connecting to remote cluster


# In[18]:


client


# ### Load Data into Dask DataFrame
# 
# **3.2** (5%) Load the data into Dask Dataframe
# - Load the dataset into Dask Dataframe
# - Use %%time to print out the loading efficiency of the operation

# In[19]:


get_ipython().run_cell_magic('time', '', "import dask.dataframe as dd\ndf_dd = dd.read_csv('minute_weather.csv')\ndf_dd.head()")


# ### Explore Data using Dask
# 
# **3.3** (5%) Summary statistics
# - Print out the shape of the dataframe
# - Print the first 10 rows of the dask dataframe
# - Print the summary statistics on all the features of the dask dataframe

# In[20]:


print('Shape of the dataframe:\n', '(', df_dd.shape[0].compute(), ',', len(df_dd.columns), ')')


# In[21]:


# Print the first 10 rows of the dask dataframe
df_dd.head(10)


# In[22]:


#Print the summary statistics on all the features of the dask dataframe
print(df_dd.describe().compute())


# ### Prepare Data using Dask 
# 
# **3.4** (5%) Data Preparation with Dask DataFrame
# - Drop the ["rowID", "hpwren_timestamp"] two columns from the dataframe
# - Perform 80/20 train and test split with `random_state=seed` (same as the previous task but in dask)
# - Print out the number of samples in train and test dataset
# 
# Note that numbers of samples are slightly difference since Dask and scikit-learn are different implementations, and also due to round-off differences.

# In[23]:


df_dd = df_dd.dropna()


# In[24]:


# Drop the ["rowID", "hpwren_timestamp"] two columns from the dataframe
df_dd = df_dd.drop(["rowID", "hpwren_timestamp"], axis = 1)


# In[25]:


# Perform 80/20 train and test split with random_state=seed (same as the previous task but in dask)
from dask_ml.model_selection import train_test_split
train_dd, test_dd = train_test_split(df_dd, test_size=0.2, random_state=seed, shuffle = True)


# In[26]:


# Print out the number of samples in train and test dataset
print('Number of samples in train dataset:\n', train_dd.shape[0].compute())
print('Number of samples in test dataset:\n', test_dd.shape[0].compute())


# **3.5** (10%) Data preprocessing and normalization with Dask
# - Perform feature normalization using the Dask library. Use only the **train** data for scaling.
# - Print out the summary statistics of the transformed features in train and test dataframes
# - Comments on your observation on the summary statistics of the transformed features in train and test dataframes

# In[27]:


# Perform feature normalization using the Dask library. Use only the train data for scaling.
from dask_ml.preprocessing import StandardScaler
scaler_1 = StandardScaler()
scalerModel_1 = scaler_1.fit(train_dd)


# In[28]:


# Print out the summary statistics of the transformed features in train and test dataframes
train_df_1 = scalerModel_1.transform(train_dd)
test_df_1 = scalerModel_1.transform(test_dd)
print(train_df_1.describe().compute())
print(test_df_1.describe().compute())


# In[29]:


# Comments on your observation on the summary statistics of the transformed features in train and test dataframes
print('By observing on the summary statistics of the transformed features in train and test dataframe, we can tell that basically both train and test dataframes have similar central tendency, dispersion and distribution.')


# ### Build Dask K-Means Model
# **3.6** (15%) KMeans clustering model with dask
# - Fit KMeans model with Dask cluster library with the transformed Dask dataframe, you should set cluster number `n_clusters` and `random_state` as the same number as previous task
# - Print out the computational performance using %%time
# 
# Note that Dask's K-Means estimator uses kmeans|| as the default algorithm.  To compare to scikit-learn's implementation of k-means, use k-means++ instead.  

# In[30]:


get_ipython().system('pip3 install --upgrade dask')


# In[31]:


get_ipython().system('pip3 install --upgrade dask[distributed]')


# In[32]:


get_ipython().run_cell_magic('time', '', '\n# Fit KMeans model with Dask cluster library with the transformed Dask dataframe, you should set cluster number n_clusters and random_state as the same number as previous task\nfrom dask_ml.cluster import KMeans\nkm = KMeans(n_clusters=9, random_state = seed, init = \'k-means++\') \n\nwith joblib.parallel_backend("dask"):\n    km.fit(train_df_1)')


# ### Evaluate Dask K-Means Model
# **3.7** (5%) Analyse hyperparameters
# - Print out the inertia_ of KMeans model
# - Print out the computational efficiency with %%time
# - Double check if the dataframes and hyperparameters are the same for both scikit-learn K-Means model and Dask K-Means model. Is the inertia_ you printed different from your answer from the previous question? Explain your observation.
# 

# **3.8** (10%) Dask K-Means estimator does not have a score() method.  As an easy fix, we can instantiate a scikit-learn K-Means estimator with the fitted Dask model (i.e., just copy the cluster centers over) to use the scikit-learn K-Means score method.
# - Print out the cluster centers found by the Dask KMeans model
# - Instantiate a scikit-learn KMeans estimator and assign the cluster centers with the one from Dask model
# - Print out the WSSE on train and test using score method. (Note that WSSE is the within-cluster sum of **square** error)

# In[33]:


get_ipython().run_cell_magic('time', '', "\n#Print out the inertia_ of KMeans model\nprint('inertia=', km.inertia_)\nprint('Yes, it is different from the previous one. This is smaller than the previous one. \\nScikit-learn uses joblib for single-machine parallelism. This lets you train most estimators using all the cores of your laptop or workstation.\\nDask registers a joblib backend. This lets you train those estimators using all the cores of your cluster , by changing one line of code.\\nThis is most useful for training large models on medium-sized datasets. ')")


# In[34]:


# Print out the cluster centers found by the Dask KMeans model
print('Cluster centers:\n', km.cluster_centers_)


# In[35]:


# Instantiate a scikit-learn KMeans estimator and assign the cluster centers with the one from Dask model
from sklearn.cluster import KMeans
km_2 = KMeans(n_clusters=9, random_state = seed)
km_2.fit(km.cluster_centers_)


# In[36]:


# Print out the WSSE on train and test using score method. (Note that WSSE is the within-cluster sum of square error)
print('WSSE_train =', -km_2.score(train_df_1))
print('WSSE_test =', -km_2.score(test_df_1))


# In[37]:


# Another way is to just assign dask_model.cluster_centers_ to sklearn_model.cluster_centers_, run sklearn_model.score on the test data to get WSSE.
kmeans.cluster_centers_ = km.cluster_centers_
print('WSSE_train =', -kmeans.score(train_df_1))
print('WSSE_test =', -kmeans.score(test_df_1))


# ### Stop the Dask Client
# 
# **3.9** (5%) Stop the dask client

# In[38]:


client.shutdown()

