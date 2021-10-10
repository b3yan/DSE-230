
************************************************************************
             TITLE: Product Recommendation and Sales Forecast
************************************************************************

For this project, we use PySpark to implement a product recommendation 
system and a sales forecast system. The main analysis tasks we used are 
the followings.

1. K-means
2. Ridge regression
3. Lasso regression
4. Gradient-boosted tree regression
5. Decision tree regression
6. Random forest regression

There are some challenges when parcing product quantities, choosing models,
and running kernel crashes due to big data. We hope we have better hardwares
And GPUs to run this faster.

************************************************************************
                            FILE MANIFEST 
************************************************************************

For EDA, please refer to files #1, #2, and #3. 
For Sales Forecast, please refer to files #4, #5, and #6. 
For Product Recommendation, please refer to files #7, #8, and #9.
File #10 is an executable script, you can run all code in this script.
File #11 is an instruction file.

1. E-commerce EDA.ipynb
2. E-commerce EDA.py
3. E-commerce EDA.pdf

4. E-commerce Sales Forecast.ipynb
5. E-commerce Sales Forecast.py
6. E-commerce Sales Forecast.pdf

7. E-commerce Product Recommendation.ipynb
8. E-commerce Product Recommendation.py
9. E-commerce Product Recommendation.pdf

10. RunAllCodeInOneScript.py

11. README.md

                              
************************************************************************
                     CONFIGURATION INSTRUCTIONS 
************************************************************************

No special configuration needed. Just make sure you have all PySpark
related packages installed.

************************************************************************
                       OPERATING INSTRUCTIONS 
************************************************************************

If you want to run any .ipynb files or .py files, just put it under the 
same folder of the E-commerce dataset. If you want to check in the .pdf 
files, simple open it and you will see all code and results.

************************************************************************
                              CONTACTS
************************************************************************

Bo Yan
b3yan@ucsd.edu

Sirish Munipalli
smunipalli@ucsd.edu

************************************************************************
                            POSSIBLE BUGs 
************************************************************************

Due to the big data size, when you run the files, you may encounter kernel
crashes. We have reduced the dataset size to reduce run time and avoid 
kernel crashes.

If you encounter java related errors, that is very likely because the data
size is too big and your CPU may crash. Don't be alarmed, you can simply 
reduce the data size and restart kernel and run all cells again. Or you 
can take a look at the PDF file for code and results.

************************************************************************
                COPYRIGHT AND LICENSING INFORMATION 
************************************************************************

All Rights Reserved.

------------------------------- END ------------------------------------
