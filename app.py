import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def load_resources():
    try:
        opening_model = joblib.load('student_performance_model_opening.pkl')
        closing_model = joblib.load('student_performance_model_closing.pkl')
        data = pd.read_csv('JEE_Rank_2016_2024.csv')
        return opening_model, closing_model, data
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None

def load_data():
    try:
        df = pd.read_csv('JEE_Rank_2016_2024.csv')  
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Define function for filtering colleges based on rank
def find_top_5_colleges(df, input_rank):
    df['Closing_Rank'] = pd.to_numeric(df['Closing_Rank'], errors='coerce')
    
    filtered_colleges = df[df['Closing_Rank'] >= input_rank]
    
    top_5_colleges = filtered_colleges.sort_values(by='Closing_Rank').head(10)
    
    return top_5_colleges


# Page 1: College Rank Filter and Prediction Section
def page_one(opening_model, closing_model, historical_data):
    st.title("IIT Seat Allocation Predictor")
    st.sidebar.header('Enter Your Details')

    institute = st.sidebar.selectbox('Institute', ['IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur'])
    academic_program = st.sidebar.selectbox('Academic Program', ['Computer Science', 'Electronics', 'Mechanical', 'Chemical', 'Civil'])
    quota = st.sidebar.selectbox('Quota', ['OPEN', 'OBC-NCL', 'SC', 'ST', 'EWS'])
    seat_type = st.sidebar.selectbox('Seat Type', ['Gender-Neutral', 'Female-Only'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    year = st.sidebar.selectbox('Year', list(range(2024, 2017, -1)))

    if st.sidebar.button('Predict Ranks', key='predict_button'):
        input_features = prepare_features(institute, academic_program, quota, seat_type, gender, year, historical_data)
        opening_rank_pred, closing_rank_pred = predict_ranks(input_features, opening_model, closing_model)
        
        if opening_rank_pred and closing_rank_pred:
            result = pd.DataFrame({
                'Institute': [institute],
                'Academic Program': [academic_program],
                'Quota': [quota],
                'Seat Type': [seat_type],
                'Gender': [gender],
                'Year': [year],
                'Predicted Opening Rank': [opening_rank_pred],
                'Predicted Closing Rank': [closing_rank_pred],
            })

            st.subheader('Prediction Results')
            st.table(result)
            create_visualizations(result, historical_data, opening_rank_pred, closing_rank_pred)

    if st.button('Next Page: College Finder'):
        st.session_state.page = "page_two"

# Page 2: College Rank Input and Filter Section
def page_two():
    df = load_data()

    if df.empty:
        st.error("Failed to load data")
        return

    st.sidebar.header("Filters")
    quotas = ['All'] + list(df['Quota'].unique())
    selected_quota = st.sidebar.selectbox('Select Quota', quotas)

    genders = ['All'] + list(df['Gender'].unique())
    selected_gender = st.sidebar.selectbox('Select Gender', genders)
    st.title("JEE Top 10 College Finder")

    input_rank = st.number_input("Enter your JEE Rank", min_value=1, max_value=100000, value=5000, step=1)

    if st.button("Find Colleges"):
        recommended_colleges = find_top_5_colleges(df, input_rank)

        if selected_quota != 'All':
            recommended_colleges = recommended_colleges[recommended_colleges['Quota'] == selected_quota]
        if selected_gender != 'All':
            recommended_colleges = recommended_colleges[recommended_colleges['Gender'] == selected_gender]

        if not recommended_colleges.empty:
            st.success(f"Found {len(recommended_colleges)} colleges for rank {input_rank}")
            st.dataframe(recommended_colleges[['Institute', 'Academic_Program_Name', 'Opening_Rank', 'Closing_Rank', 'Quota', 'Gender', 'Seat_Type']])

            st.subheader("Opening Ranks Comparison")
            chart_data = recommended_colleges.set_index('Institute')['Opening_Rank']
            st.bar_chart(chart_data)
        else:
            st.warning("No colleges found for the given criteria.")

    if st.button('Back to Rank Prediction'):
        st.session_state.page = "page_one"

def prepare_features(institute, academic_program, quota, seat_type, gender, year, historical_data):
    features = {
        'Institute': 0, 'Quota': 0, 'Gender': 0, 'Year': year, 'Academic_Program_Name': 0, 'Seat_Type': 0,
    }
    
    if historical_data is not None:
        hist_ranks = historical_data[
            (historical_data['Institute'] == institute) &
            (historical_data['Academic_Program_Name'] == academic_program)
        ]
        if not hist_ranks.empty:
            avg_opening = hist_ranks['Opening_Rank'].mean()
            avg_closing = hist_ranks['Closing_Rank'].mean()

        features['Institute'] = historical_data[historical_data['Institute'] == institute].index[0] if not historical_data[historical_data['Institute'] == institute].empty else 0
        features['Quota'] = historical_data[historical_data['Quota'] == quota].index[0] if not historical_data[historical_data['Quota'] == quota].empty else 0
        features['Gender'] = historical_data[historical_data['Gender'] == gender].index[0] if not historical_data[historical_data['Gender'] == gender].empty else 0
        features['Academic_Program_Name'] = historical_data[historical_data['Academic_Program_Name'] == academic_program].index[0] if not historical_data[historical_data['Academic_Program_Name'] == academic_program].empty else 0
        features['Seat_Type'] = historical_data[historical_data['Seat_Type'] == seat_type].index[0] if not historical_data[historical_data['Seat_Type'] == seat_type].empty else 0

    return pd.DataFrame([features])

def predict_ranks(input_features, opening_model, closing_model):
    try:
        closing_rank_pred = int(opening_model.predict(input_features)[0])
        opening_rank_pred = int(closing_model.predict(input_features)[0])
        return opening_rank_pred, closing_rank_pred
    except Exception as e:
        st.error(f"Error predicting ranks: {str(e)}")
        return None, None

# Visualize the data
def create_visualizations(result, historical_data, opening_rank_pred, closing_rank_pred):
    st.subheader("Trend Analysis and Visualizations")
    
    historical_data['Opening_Rank'] = pd.to_numeric(historical_data['Opening_Rank'], errors='coerce')
    historical_data['Closing_Rank'] = pd.to_numeric(historical_data['Closing_Rank'], errors='coerce')
    
    # 1. Year-wise Rank Trends
    year_trends = historical_data.groupby('Year').agg({
        'Opening_Rank': 'mean',
        'Closing_Rank': 'mean'
    }).reset_index()
    
    fig_year = px.line(year_trends, x='Year', y=['Opening_Rank', 'Closing_Rank'],
                       title='Year-wise Rank Trends',
                       labels={'value': 'Rank', 'variable': 'Rank Type'})
    st.plotly_chart(fig_year)

    # 2. Institute-wise Average Ranks
    inst_ranks = historical_data.groupby('Institute').agg({
        'Opening_Rank': 'mean',
        'Closing_Rank': 'mean'
    }).reset_index()
    
    fig_inst = px.bar(inst_ranks, x='Institute', y=['Opening_Rank', 'Closing_Rank'],
                      title='Institute-wise Average Ranks',
                      barmode='group',
                      labels={'value': 'Average Rank', 'variable': 'Rank Type'})
    st.plotly_chart(fig_inst)

    # 3. Program Popularity (based on number of applications)
    prog_pop = historical_data['Academic_Program_Name'].value_counts().head(10).reset_index()
    prog_pop.columns = ['Program', 'Count']
    
    fig_prog = px.pie(prog_pop, values='Count', names='Program',
                      title='Top 10 Programs Distribution')
    st.plotly_chart(fig_prog)

    # 4. Gender Distribution Analysis
    gender_ranks = historical_data.groupby(['Gender', 'Academic_Program_Name'])['Opening_Rank'].mean().reset_index()
    
    fig_gender = px.box(historical_data, x='Gender', y='Opening_Rank',
                       color='Academic_Program_Name',
                       title='Gender-wise Rank Distribution by Program')
    st.plotly_chart(fig_gender)

    # 5. Quota-wise Analysis
    quota_ranks = historical_data.groupby('Quota').agg({
        'Opening_Rank': 'mean',
        'Closing_Rank': 'mean'
    }).reset_index()
    
    fig_quota = px.bar(quota_ranks, x='Quota', y=['Opening_Rank', 'Closing_Rank'],
                       title='Quota-wise Average Ranks',
                       barmode='group',
                       labels={'value': 'Average Rank', 'variable': 'Rank Type'})
    st.plotly_chart(fig_quota)


def app():
    if 'page' not in st.session_state:
        st.session_state.page = "page_one"

    opening_model, closing_model, historical_data = load_resources()

    if opening_model and closing_model and historical_data is not None:
        if st.session_state.page == "page_one":
            page_one(opening_model, closing_model, historical_data)
        elif st.session_state.page == "page_two":
            page_two()

if __name__ == "__main__":
    app()
