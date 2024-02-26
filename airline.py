import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Set page title and icon
st.set_page_config(page_title="Airline Passenger Satisfaction ", page_icon=":airplane:")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Modeling", "Make Predictions!"])

df = pd.read_csv('data/df.csv')

# Home Page
if page == "Home":
    st.title("Airline Passenger Satisfaction Prediction Project:airplane:")
    st.subheader("Welcome to my Airline Satisfaction Prediction App!")
    st.write("This app is designed to make the exploration and analysis of the Airline Passenger Satisfaction Prediction Dataset easy and accessible. Whether you're interested in the distribution of data, relationships between features, or the performance of a machine learning model, this tool provides an interactive and visual platform for your exploration. With this model, you will be able to easily make predictions on wether the passenger belonging to an airline was satisfied or neutral/dissatisfied.")
    st.image('https://media.licdn.com/dms/image/C4D12AQEYUpGT_USmeQ/article-cover_image-shrink_720_1280/0/1602352478948?e=1714003200&v=beta&t=c_HI1hIwQlxpEiX-t9PpcQ8a912MSlRmuj8_BI9fctE')
    st.write("Use the sidebar to navigate between different sections.")

# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("The dataset contains information about airline passengers and their flight details, including features such as their gender,flight distance, seat comfort, inflight entertainment, seat comfort, and more. Given this information we are able to see whether or not a passenger was satsfied or not with the airline.")
    st.image('https://media.licdn.com/dms/image/D5612AQG9UOiKKMdsyQ/article-cover_image-shrink_720_1280/0/1675107387553?e=2147483647&v=beta&t=2JWEK7R60Il22O9ZWVLRuznZtxGjVhYfgu_WwKUQ_yk')
    st.link_button("Click Here for The Airline Passenger Satisfaction Kaggle Dataset", "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data", help = None)



    st.subheader("Sneak Peak at the Data")

    # Display dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df)
    
    # Column List
    if st.checkbox("Column List"):
        st.code(f"Columns: {df.columns.tolist()}")
        if st.toggle("Further breakdown of columns"):
            num_cols = df.select_dtypes(include='number').columns.tolist()
            obj_cols = df.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols}\nObject Columns: {obj_cols}")

    # Shape
    if st.checkbox("Shape"):
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")



elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    eda_type = st.multiselect("What type of EDA are you interested in exploring?", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    obj_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()


    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for your histogram:", num_cols, index = None)
        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue on Histogram"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, title = chart_title, color = 'satisfaction', barmode = 'overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title = chart_title))
    
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for your box plot:", num_cols, index = None)
        if b_selected_col:
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue on Box Plot"):
                st.plotly_chart(px.box(df, x=b_selected_col, y = 'satisfaction', title = chart_title, color = 'satisfaction'))
            else:
                st.plotly_chart(px.box(df, x=b_selected_col, title = chart_title))



            
    
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index = None)

        if selected_col_x and selected_col_y:
            hue_toggle = st.toggle("Satisfaction Hue on Scatterplot")
            chart_title = f"{' '.join(selected_col_x.split('_')).title()} vs. {' '.join(selected_col_y.split('_')).title()}"

            if hue_toggle:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='satisfaction', title = chart_title))
            else:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color = 'satisfaction', title = chart_title))
        
    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", obj_cols, index = None)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, title = chart_title, color = 'satisfaction'))
            
if page == "Modeling":
    st.title(':gear: Modeling')
    st.markdown("On this page, you can see how well different *machine learning models* make predictions on the Airline Passenger Satisfaction Dataset. The best machine learning model is the one that beats the baseline accuracy score which is 56.7%.")
    
    # Set up X and Y
    features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
             
    X = df[features]
    y = df['satisfaction']
    
    
    #Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    
    model_option = st.selectbox('Select a Model:', ['Logistic Regression', 'Random Forest', 'KNN'], index = None)
    
    if model_option: 
        if model_option == 'Logistic Regression':
            model = LogisticRegression()
        elif model_option == 'Random Forest':
            model = RandomForestClassifier()
        elif model_option == 'KNN':
            k_value = st.slider('Select a number of k', 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors =k_value)
            
        if st.button("Let's see the performance!"): 
            
            
            model.fit(X_train_sc, y_train)
            
            # Display results
            st.subheader(f'{model} Evaluation:')
            st.text(f'Training Accuracy: {round(model.score(X_train_sc, y_train),3)}')
            st.text(f'Testing Accuracy: {round(model.score(X_test_sc, y_test),3)}')
            
            # Add section for Confusion Matrix
            ConfusionMatrixDisplay.from_estimator(model, X_test_sc, y_test, cmap = 'Blues')
            
            #This is turning confusion matrix onto display 'get current figure'
            cm_fig = plt.gcf()
            
            st.pyplot(cm_fig)
            
            st.write('Compared to our baseline of 56.7% we can see that our model beat the baseline!')



if page == "Make Predictions!":
    st.title(":rocket: Make Predictions on Airline Passenger Satisfaction!")

    # Create sliders for user to input data
    st.subheader("Adjust the sliders to input data:")

    s_l = st.slider("Age", 1.0, 100.0, 1.0, 1.0)
    s_w = st.slider("Flight Distance", 0.01, 10.0, 0.01, 0.01)
    p_l = st.slider("Inflight wifi service", 0.01, 10.0, 0.01, 0.01)
    p_w = st.slider("Departure/Arrival time convenient", 0.01, 10.0, 0.01, 0.01)
    st.slider("Ease of Online booking", 0.01, 10.0, 0.01, 0.01)
    st.slider("Gate location", 0.01, 10.0, 0.01, 0.01)
    st.slider("Food and drink", 0.01, 10.0, 0.01, 0.01)
    st.slider("Online boarding", 0.01, 10.0, 0.01, 0.01)
    st.slider("Seat comfort", 0.01, 10.0, 0.01, 0.01)
    st.slider("Inflight entertainment", 0.01, 10.0, 0.01, 0.01)
    st.slider("Leg room service", 0.01, 10.0, 0.01, 0.01)
    st.slider("Baggage handling", 0.01, 10.0, 0.01, 0.01)
    st.slider("Checkin service", 0.01, 10.0, 0.01, 0.01)
    st.slider("Inflight service", 0.01, 10.0, 0.01, 0.01)
    st.slider("Cleanliness", 0.01, 10.0, 0.01, 0.01)
    st.slider("Departure Delay in Minutes", 0.01, 10.0, 0.01, 0.01)
    st.slider("Arrival Delay in Minutes", 0.01, 10.0, 0.01, 0.01)

    # Your features must be in order that the model was trained on
    user_input = pd.DataFrame({
            'sepal_length': [s_l],
            'sepal_width': [s_w],
            'petal_length': [p_l],
            'petal_width': [p_w]
            })

    # Check out "pickling" to learn how we can "save" a model
    # and avoid the need to refit again!
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[features]
    y = df['species']

    # Model Selection
    st.write("The predictions are made using KNN as it performed the best out of all of the models.")
    model = KNeighborsClassifier()
        
    if st.button("Make a Prediction!"):
        model.fit(X, y)
        prediction = model.predict(user_input)
        st.write(f"{model} predicts this airline passenger is {prediction[0]}!")
        st.balloons()