#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[2]:


data_fake=pd.read_csv(r'C:\Users\Aswini\Downloads\Fake.csv')
data_true=pd.read_csv(r'C:\Users\Aswini\Downloads\True.csv')


# In[3]:


data_fake.head()


# In[4]:


data_true.head()


# In[5]:


data_fake['class']=0
data_true['class']=1


# In[6]:


data_fake.shape


# In[7]:


data_true.shape


# In[8]:


data_fake_manual_testing=data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i],axis=0,inplace=True)
data_true_manual_testing=data_fake.tail(10)
for i in range(21416,21406,-1):
    data_fake.drop([i],axis=0,inplace=True)


# In[9]:


data_fake_manual_testing['class']=0
data_true_manual_testing['class']=1


# In[10]:


data_fake_manual_testing.head(10)


# In[11]:


data_true_manual_testing.head(10)


# In[12]:


data_merge=pd.concat([data_true,data_fake],axis=0)
data_merge.head(10)


# In[13]:


data_merge.columns


# In[14]:


data=data_merge.drop(['subject','title','date'],axis=1)
data


# In[15]:


data.isnull().sum()


# In[16]:


data=data.sample(frac=1)
data


# In[17]:


data.columns


# In[18]:


data.head()


# In[19]:


import re
import string
def wordpot(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation + '’‘“”'), '', text)  # Removes curly quotes too
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)  # Removes words with numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# In[ ]:


data['text']=data['text'].apply(wordpot)


# In[21]:


data['text'].isnull().sum()


# In[22]:


data.head()


# In[23]:


x=data['text']
 
y=data['class']


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
print(x_train[:5])
print(x_train.head())


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)


# In[26]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xv_train,y_train)


# In[27]:


pred_lr=LR.predict(xv_test)


# In[28]:


LR.score(xv_test,y_test)


# In[29]:


print(classification_report(y_test,pred_lr))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)


# In[31]:


predict_dt=DT.predict(xv_test)


# In[32]:


DT.score(xv_test,y_test)


# In[33]:


print(classification_report(y_test,predict_dt))


# In[34]:


from sklearn.ensemble import GradientBoostingClassifier
GB=GradientBoostingClassifier(random_state=0)
GB.fit(xv_train,y_train)


# In[ ]:


predict_GB=GB.predict(xv_test)


# In[36]:


GB.score(xv_test,y_test)


# In[37]:


print(classification_report(y_test,predict_dt))


# In[39]:


from  sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(random_state=0)
RF.fit(xv_train,y_train)


# In[40]:


predict_RF=RF.predict(xv_test)


# In[41]:


RF.score(xv_test,y_test)


# In[42]:


print(classification_report(y_test,predict_RF))


# In[50]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}  
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)  
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)  
    pred_GB = GB.predict(new_xv_test)  
    pred_RF = RF.predict(new_xv_test)  
    

    print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_lable(pred_LR[0]),
        output_lable(predict_dt[0]),
        output_lable(predict_GB[0]),
        output_lable(predict_RF[0])
    ))


# In[ ]:


news=str(input())
manual_testing(news)


# In[ ]:




