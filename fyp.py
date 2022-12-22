import streamlit as st 
import numpy as np
import plotly.express as px 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

# Make container 
header = st.container()
data_sets = st.container()
model_training = st.container()

with header: 
    st.title("Final Year Project App")
    st.text("We will work with parameter dataset of Induction Motor collected through sensors")
    
with data_sets:
    st.header("Parameter Dataset")
  
    #import data 
    df = pd.read_csv("Dataset2.csv")
    df = df.dropna()
    st.write(df.head(10))
    # st.write(df.columns)
    # lis = []
    # for i in df.columns:
    #     lis.append(i)
    # st.write(lis)
    # st.write(df['Month '].to_string(index=False))
    
    st.write("The shape of the Dataset: ", df.shape)
    
    fig = px.scatter(df, x="TESTING AMP", y="VOLT", hover_name="Location", color="Location",
     width=None, height=None)
    st.write(fig)
     
    year_option = df['Year '].unique().tolist()
    Years = st.selectbox("You can see the Fault conditions with rpm per Month ", year_option,0)
    
    fig1 = px.scatter(df , x="Location", y="RPM", color="Location", hover_name="Month ",animation_group="Location",
                      animation_frame='Month ')
    st.write(fig1)
    
    fig2 = px.scatter(df , x="Location", y="VOLT", color="Location", hover_name="Month ",animation_group="Location")
    st.write(fig2)
    
    fig3 = px.scatter(df , x="Location", y="TESTING AMP", color="Location", hover_name="Month ",animation_group="Location")
    st.write(fig3)
    
    
with model_training:
    
    st.header("Machine Learning Algorithm Results")
    
    features , lables = st.columns(2)
    with features:
         st.text("These are the features used for ML")
         X = df[["VOLT", "TESTING AMP", "RPM"]]
         st.write(X)
         
    with lables:    
         st.text("These are the lables used for ML")
         y = df[["Location"]]
         st.write(y)
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model = model.fit(x_train, y_train)
    
    future, accuracy = st.columns(2)
    
    with future:
        st.subheader("ML Result")
        a = st.number_input("Input a value of Volatge" ,min_value=200, max_value=450)
        b = st.number_input("Input a value of Testing Current ", min_value=1, max_value=12)
        c = st.number_input("Input a value of Motor RPM",min_value=1500, max_value=3000)
        
        predictions = model.predict([[a,b,c]])
        st.write("This is the prediction: ",predictions)
        
    with accuracy:
        st.subheader("Predicrions based upon the ML model")
        predictions = model.predict(x_test)
        st.write(predictions)
        
        st.subheader("Accuracy Score Result")
        accuracy = model.score(x_test,y_test)
        st.write('Score for Training data = ', accuracy)
        
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, predictions)
        st.write('Score for Training data = ',score)

from sklearn import metrics
from sklearn.metrics import confusion_matrix

st.subheader("Confusion Matrix")
cm = metrics.confusion_matrix(y_test, predictions)
cm      
plt.figure(figsize=(250,250))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, 
            square=True, cmap= 'Spectral')
plt.ylabel('Actual Output')
plt.xlabel('predicted Output')
all_sample_title = 'Accuracy score: {0}'.format(score)
st.write(plt.title(all_sample_title, size=15))
        
        
    
    