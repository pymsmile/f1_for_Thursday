# -*- coding: utf-8 -*-
"""
Project Name: Application for Data Scientist in Nearmap (Technology – Barangaroo, New South Wales)
Task:Calculate the f1 for Thursdays... this is not a trick question! We will search for the first 5 d.p. in your CV
Answer: f1 for Thursdays = 0.307692
Created on 06/03/2019
@author: Yanming Pei
------------------------------------------------------
Copyright (c) <2019> <Yanming PEI>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import pandas as pd
from sklearn.metrics import f1_score
# %% Read file
i_file = 'C:/Users/ypei/Documents/PythonCodes/Nearmap -Job/Data Scientist/test.psv'
dates_df = pd.read_csv(i_file, skiprows=1)

# %% Clean the data
dates_df = dates_df.iloc[:,0].str.split('|',expand=True)
dates_df.rename(columns={0:'date',1:'y',2:'yhat'}, inplace=True)

# %% Filter the Thursday
dates_df['date'] = pd.to_datetime(dates_df['date'])
dates_df['day_of_week'] = dates_df.loc[:,'date'].dt.day_name()
thursday_df = dates_df[dates_df['day_of_week'].str.contains('Thursday')]

# %% Data integrity check
any_null_check = thursday_df.isnull().values.any()
thursday_df.head()

# %% Method 1:
# F1 score definition: https://en.wikipedia.org/wiki/F1_score
y_true = thursday_df.loc[:,'y'].astype(int)
y_pred = thursday_df.loc[:,'yhat'].astype(int)
num_data = len(y_true)

true_pos = sum((y_true==1) & (y_pred==1))
false_pos = sum((y_true==0) & (y_pred==1))
false_neg = sum((y_true==1) & (y_pred==0))
true_neg = sum((y_true==0) & (y_pred==0))

precision = true_pos / (true_pos+false_pos)
recall = true_pos / (true_pos+false_neg)

f1_defination = 2 * (precision*recall) / (precision+recall)
print('f1 for Thursdays using definition = %.6f' %f1_defination) # 0.307692

# %% Method 2: Use sklearn.metrics.f1_score
# F1 score definition: https://en.wikipedia.org/wiki/F1_score

f1_sklearn = f1_score(y_true, y_pred, average=None)
print('f1 for Thursdays using sklearn = %.6f' %f1_sklearn[1]) # 0.307692

# %%
print('Thank you very much. \n\nRegards,\nYanming')
