#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import streamlit as st
import joblib


# In[3]:


x_numericos = {'host_listings_count':0 , 'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 
               'beds': 0, 'extra_people': 0, 'Ano': 0, 'numero_amenities': 0}

x_tf = {'instant_bookable': 0}


# In[4]:


for item in x_numericos:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.00000, format="%.5f")
    elif item == 'extra_people':
        valor = st.number_input(f'{item}', step=0.01, value=0.0)
    else:
        valor = st.number_input(f'{item}', step=1, value=0)
        
    x_numericos[item] = valor
    
    
for item in x_tf:
    valor = st.selectbox(f'{item}', ('Sim', 'Não'))
    if valor =='Sim':
        x_tf[item] = 1
    else:
        x_tf[item] = 0
    
botao = st.button('Calcular valor da diária')

dicionario = {}

if botao:
    dicionario.update(x_numericos)
    dicionario.update(x_tf)
    valores_x = pd.DataFrame(dicionario, index=[0])
    modelo = joblib.load('modelo.joblib')
    preco = modelo.predict(valores_x)
    
    st.write(preco)
    

