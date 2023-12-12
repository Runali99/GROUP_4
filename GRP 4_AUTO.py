#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
     


# In[2]:


import pandas as pd


# In[3]:


df_train=pd.read_csv("C://Users//hp//Downloads//Train.csv")


# In[7]:


df_test=pd.read_csv("C://Users//hp//Downloads//Test.csv")


# In[5]:


df_train.isna().sum()


# In[8]:


df_train_filled = df_train.apply(lambda x: x.fillna(x.mode()[0]))


# In[9]:


df_train_filled.head()


# In[10]:


df_train_filled.isna().sum()


# In[11]:


from sklearn.preprocessing import StandardScaler
sc =StandardScaler


# In[12]:


from sklearn.preprocessing import StandardScaler


selected_columns = ['Age', 'Work_Experience', 'Family_Size']
scaler = StandardScaler()

df_train_filled[selected_columns] = scaler.fit_transform(df_train_filled[selected_columns])

     


# In[13]:


df_train_filled.head()


# In[16]:


from sklearn.preprocessing import LabelEncoder
b=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Var_1","Segmentation"]
le = LabelEncoder()
for i in b:
 df_train_filled[i]=le.fit_transform(df_train_filled[i])


# In[22]:


df_train_filled.head()


# In[23]:


df_train_filled.columns


# In[25]:


x=df_train_filled[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size','Var_1']]


# In[26]:


Y=df_train_filled['Segmentation']


# In[27]:


from sklearn.tree import DecisionTreeClassifier
dt =  DecisionTreeClassifier()
dt = dt.fit(x,Y)
    


# In[30]:


dt.fit(x,Y)
     


# In[ ]:





# In[34]:


df_TEST_filled = df_test.apply(lambda x: x.fillna(x.mode()[0]))


# In[35]:


from sklearn.preprocessing import LabelEncoder
b=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Var_1","Segmentation"]
le = LabelEncoder()
for i in b:
 df_TEST_filled[i]=le.fit_transform(df_TEST_filled[i])
     


# In[41]:


x_TEST=df_TEST_filled[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size','Var_1']]
     


# In[42]:


y_TEST=df_TEST_filled['Segmentation']


# In[43]:


Y_PRED=dt.predict(x_TEST)


# In[44]:


from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score

p = precision_score(y_TEST, Y_PRED, average='micro')
r = recall_score(y_TEST, Y_PRED, average='micro')
a = accuracy_score(y_TEST, Y_PRED)
f1 = f1_score(y_TEST, Y_PRED, average='micro')


# In[45]:


p


# In[ ]:




