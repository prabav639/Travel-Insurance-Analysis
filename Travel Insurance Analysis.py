#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score


# In[2]:


insurance=pd.read_csv('/Users/prabavmurali/Downloads/travel insurance.csv')


# In[3]:


insurance.head()


# In[4]:


insurance.describe()


# In[5]:


insurance.rename(columns={'Commision (in value)':'Commission'},inplace=True)


# In[6]:


insurance.isna().sum()


# In[7]:


insurance['Gender'].unique()


# In[8]:


insurance=insurance.drop(['Gender'],axis=1)


# In[9]:


column_keys=insurance.select_dtypes(include=['object']).columns.tolist()
for key in column_keys:
    print('Unique elements of',key,'are: ')
    print(insurance[key].unique(),end='\n')
    print(end='\n')


# In[10]:


import plotly.express as exp
dest = insurance.groupby(by=["Destination"]).size().reset_index(name="counts")
dest.nlargest(15,['counts'])
dest['DestinationNew'] = np.where(dest['counts']>1500, dest['Destination'], 'Others')
fig = exp.pie(dest,values='counts', names=dest['DestinationNew'], title='Destination most insured', hole=0.3)
fig.show()


# In[11]:


name = insurance.groupby(by=["Product Name"]).size().reset_index(name="counts")
name.nlargest(15,['counts'])
name['ProdName'] = np.where(name['counts']>1000, name['Product Name'], 'Others')
fig = exp.pie(name, values='counts', names=name['ProdName'], title='Plans opted by Customers', hole=0.3)
fig.show()


# In[24]:


fig = plt.figure(figsize = (10, 5))
plt.hist(insurance['Age'],color='turquoise')
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.title("Distribution of Age")
plt.show()


# In[14]:


num=insurance.groupby(by=['Agency Type']).size().reset_index(name="counts")
plt.bar(num['Agency Type'], num['counts'], color ='orange',width = 0.4)
plt.xlabel("Agency Type")
plt.ylabel("Number of people insured")
plt.title("People insured under different Agency Type")
plt.show()


# In[15]:


from sklearn import preprocessing
lbla = preprocessing.LabelEncoder()
insurance['Agency']= lbla.fit_transform(insurance['Agency'])
insurance['Agency Type']= lbla.fit_transform(insurance['Agency Type'])
insurance['Distribution Channel']= lbla.fit_transform(insurance['Distribution Channel'])
insurance['Product Name']= lbla.fit_transform(insurance['Product Name'])
insurance['Claim']= lbla.fit_transform(insurance['Claim'])
insurance['Destination']= lbla.fit_transform(insurance['Destination'])


# In[16]:


Agesect=[]
for i in insurance['Age']:
    if i<=30:
        Agesect.append(1)
    elif i>=61:
        Agesect.append(3)
    else:
        Agesect.append(2)
insurance['Agegroup']=Agesect
insurance


# In[17]:


x=pd.DataFrame()
y=pd.DataFrame()
y['Claimed']=insurance['Claim']
x=insurance.drop(['Claim','Age'],axis=1)


# In[18]:


sm = SMOTE(random_state=42)
x_sm, y_sm = sm.fit_resample(x, y)
y_sm.value_counts(normalize=True) * 100


# In[19]:


x_new=pd.DataFrame()
to_scale = x_sm.columns
scaler = MinMaxScaler()
x_new[to_scale] = scaler.fit_transform(x_sm[to_scale])
x_new.head()


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x_new, y_sm, test_size=0.20, random_state=1)


# In[21]:


rfc = RandomForestClassifier(random_state=2)
rfc.fit(x_train, np.ravel(y_train))
pred = rfc.predict(x_test)


# In[22]:


print(f'Accuracy score is: {accuracy_score(y_test, pred):.2f}\nRecall score is: {recall_score(y_test, pred):.2f}\n')
print('F1 score is :',f1_score(y_test, pred, average='macro'))


# In[ ]:




