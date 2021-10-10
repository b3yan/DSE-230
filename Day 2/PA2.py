#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
pathToFile = os.environ['DATA_FILE_PATH']


# ## Please notice that I put the task 3 at the end of the notebook!

# ### Task 1 – Remove punctuation and print the first 25 rows (2 points)

# #### 1.1 (From JupyterLab terminal) Copy data file(BookReviews_1M.txt) from local FS to HDFS

# In[1]:


get_ipython().system('pwd')


# In[2]:


get_ipython().system('ls')


# In[3]:


get_ipython().system('hadoop fs -mkdir /hdfs_folder1')


# In[4]:


get_ipython().system('hadoop fs -copyFromLocal BookReviews_1M.txt /hdfs_folder1')


# #### 1.2 Start spark session

# In[5]:


import pyspark
from pyspark.sql import SparkSession
import wordcount_utils
import pyspark.sql.functions as f
import time
import numpy as np


# In[6]:


#conf = pyspark.SparkConf().setAll([('spark.master', 'local[1]'), ('spark.app.name', 'Word Count')])
#conf = pyspark.SparkConf().setAll([('spark.master', 'local[2]'), ('spark.app.name', 'Word Count')])
conf = pyspark.SparkConf().setAll([('spark.master', 'local[4]'), ('spark.app.name', 'Word Count')])


# In[7]:


spark = SparkSession.builder.config(conf = conf).getOrCreate()


# #### 1.3 Read data from HDFS into Spark DataFrame

# In[8]:


df = spark.read.text('/hdfs_folder1/BookReviews_1M.txt')


# #### 1.4 Remove punctuations using the removePunctuation function imported from wordcount_utils.py. Hint – Pass the "value" column to removePunctuation

# In[9]:


book_reviews_df = (df.select(wordcount_utils.removePunctuation('value')))


# #### 1.5 Print the first 25 rows of the resulting dataframe

# In[10]:


print(book_reviews_df.show(25, truncate = True))


# ### Task 2 – Find and print top 100 words based on count (5 points)

# #### 2.1 Split each sentence(row) into words based on the delimited space (" ")

# In[11]:


words_df = (book_reviews_df.select(f.split(book_reviews_df.sentence, ' ').alias('words')))


# #### 2.2 Put each word in each sentence into its own rows and assign the result to a new dataframe

# In[12]:


word_df = (words_df.select(f.explode(words_df.words).alias('word')))


# #### 2.3 Remove all empty lines in the dataframe (due to empty lines and/or words)

# In[13]:


clean_word_df = (word_df.where(word_df.word != ''))


# #### 2.4 Group rows in the dataframe by unique words and count the rows in each group

# In[14]:


word_count = clean_word_df.groupBy("word").count()


# #### 2.5 Sort the word count dataframe

# In[15]:


sorted_word_count = (word_count.orderBy('count', ascending = 0))


# #### 2.6 Print the first 100 rows of the dataframe sorted by word count

# In[16]:


print(sorted_word_count.show(100, truncate = True))


# ### Task 4 – Save the sorted word counts to HDFS as a CSV file (1 point)

# #### 4.1 Coalesce the sorted dataframe(task 2) to one partition. This makes sure that all our results will end up in the same CSV file. Save the 1-partition dataframe to HDFS using the DataFrame.write.csv() method.

# In[17]:


sorted_word_count.coalesce(1).write.csv("hdfs:///wordCountsSorted.csv", header = True, mode = "overwrite")


# #### 4.2 Copy the file from HDFS to local file system:

# • Run hadoop fs -ls / to list the root directory of the HDFS. You should see the CSV file that you have saved. Counterintuitively, this CSV file is a folder, which contains individually saved files from each partition of the saved dataframe

# In[18]:


get_ipython().system('hadoop fs -ls /')


# • Run hadoop fs -ls /wordCountsSorted.csv/ to see what is inside the saved folder. Since we made sure to coalesce our dataframe to just one partition, we should expect to find only one saved partition in this folder, saved also as a CSV. Note the name of this file, it should look something like part-00000-xx.....xx.csv.

# In[19]:


get_ipython().system('hadoop fs -ls /wordCountsSorted.csv/')


# • Run the following command to copy the results CSV from HDFS to the current folder on your local file system. Rename it to something simple like results.csv :
# hadoop fs -copyToLocal /wordCountsSorted.csv/part-00000-*.csv .

# In[20]:


get_ipython().system('hadoop fs -copyToLocal /wordCountsSorted.csv/part-00000-5e673968-879a-40f2-a430-12f57805087a-c000.csv')


# • We want you to submit a CSV containing the first 101 rows of the results file(header row + top 100 words by count). To do this, use the command (You can also do so manually since CSV files are in plain text):
# head -n 101 results.csv > 101_rows.csv

# In[23]:


get_ipython().system('head -n 101 results.csv > 101_rows.csv')


# #### Stop spark session

# In[24]:


spark.stop()


# ### Task 3 – Record execution time for 1, 2 and 4 cores (2 points)### Task 3 – Record execution time for 1, 2 and 4 cores (2 points)
# * Repeat task 1 and 2 for 1, 2 and 4(if available) cores.
# * Repeat the experiment 3 times for each case and record execution time - from reading the data into a dataframe to printing the top 100 words. Hint – use time.time() in python
# * Record execution time in the below format and save it to execution_time.csv:
# * #cores, time0, time1, time2, mean, stdev

# In[25]:


cores = [1, 2, 4]

for i in cores:
    print(i, 'cores')
    execution_time = []
    import pyspark
    from pyspark.sql import SparkSession
    import wordcount_utils
    import pyspark.sql.functions as f
    import time
    import numpy as np
    conf = pyspark.SparkConf().setAll([('spark.master', 'local[%s]' % i), ('spark.app.name', 'Word Count')])   
    
    for j in range(3):
        import time
        spark = SparkSession.builder.config(conf = conf).getOrCreate()
        start_time = time.time()
        df = spark.read.text('/hdfs_folder1/BookReviews_1M.txt')
        book_reviews_df = (df.select(wordcount_utils.removePunctuation('value')))
        print(book_reviews_df.show(25, truncate = True))
        words_df = (book_reviews_df.select(f.split(book_reviews_df.sentence, ' ').alias('words')))
        word_df = (words_df.select(f.explode(words_df.words).alias('word')))
        clean_word_df = (word_df.where(word_df.word != ''))
        word_count = clean_word_df.groupBy("word").count()
        sorted_word_count = (word_count.orderBy('count', ascending = 0))
        print(sorted_word_count.show(100, truncate = True))
        end_time = time.time()
        time = end_time - start_time
        execution_time.append(time)
        print('time', j, '=', time)        
        spark.stop()
    m = np.mean(execution_time)
    s = np.std(execution_time)
    print('mean =', m)
    print('stdev =', s)
    print('\n')


# In[ ]:




