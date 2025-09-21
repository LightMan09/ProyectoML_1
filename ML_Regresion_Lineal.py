#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression


# In[ ]:


#Cargar data set 
df = pd.read_csv("Housing.csv").drop("prefarea", axis=1)

df_train = df.drop("price", axis=1)
df_test = df["price"]


# In[55]:


X_train, X_test, Y_train, Y_test = train_test_split(df_train, df_test, test_size=0.25, random_state=50)


# In[18]:


#Transformar una columna de string a varias con un identificador de cada string diferente
#de la columna
#Cuando no tiene orden jerarquico 
encoderOneHot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform = "pandas")
encodeOH_X_train = encoderOneHot.fit_transform(X_train[["furnishingstatus"]])
encodeOH_X_test = encoderOneHot.transform(X_test[["furnishingstatus"]])



# In[19]:


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


# In[22]:


#Escalar todo el data set 
scaler = StandardScaler()
X_train_scaler = pd.DataFrame(scaler.fit_transform(X_train), columns = X_test.columns, index = X_train.index)
X_test_scaler = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)


# In[82]:


#Regresion lineal 

LinReg= LinearRegression()

LinReg.fit(X_train_scaler, Y_train)
predict = LinReg.predict(X_test_scaler).round(0)
predict = pd.DataFrame(predict, index = Y_test.index, columns = ["pricePredict"])
Y_test = pd.DataFrame(Y_test)
new = pd.concat([predict, Y_test], axis=1)


# In[81]:


#Calcular la presicion del modelo
from sklearn.metrics import  r2_score


r2 = r2_score(Y_test, predict)
print("R²:", r2)

