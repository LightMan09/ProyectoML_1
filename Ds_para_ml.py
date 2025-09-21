#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# In[3]:


#Cargar data set 
df = pd.read_csv("Housing.csv").drop("prefarea", axis=1)

df_train = df.drop("price", axis=1)
df_test = df["price"]


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(df_train, df_test, test_size=0.25, random_state=50)


# In[5]:


#Transformar una columna de string a varias con un identificador de cada string diferente
#de la columna
#Cuando no tiene orden jerarquico 
encoderOneHot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform = "pandas")
encodeOH_X_train = encoderOneHot.fit_transform(X_train[["furnishingstatus"]])
encodeOH_X_test = encoderOneHot.transform(X_test[["furnishingstatus"]])


# In[6]:


#Transformar columnas en numeros, con un orden jerarquico

#las columnas que se trabajara
colu = [
    "mainroad", 
    "guestroom", 
    "basement",
    "hotwaterheating",
    "airconditioning"
]

#Creaccion del encoderOrdinal con cuales strings trabajara (yes, no) por el numero de columnas en las que 
#hara el cambio
encoderOrdinal = OrdinalEncoder(categories = [["yes", "no"]] * len(colu))


X_train[colu] = encoderOrdinal.fit_transform(X_train[colu])
X_test[colu] = encoderOrdinal.transform(X_test[colu])


X_train = pd.concat([X_train, encodeOH_X_train], axis = 1).drop(columns = "furnishingstatus")
X_test = pd.concat([X_test, encodeOH_X_test], axis = 1).drop(columns = "furnishingstatus")


# In[7]:


#Escalar todo el data set 
scaler = StandardScaler()
X_train_scaler = pd.DataFrame(scaler.fit_transform(X_train), columns = X_test.columns, index = X_train.index)
X_test_scaler = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)

X_train_scaler.head()

