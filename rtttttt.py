#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title of the Application
st.title("User Behavior Analysis and Insights")

# Markdown: THE STORYLINE / INSIGHTS
st.markdown("# THE STORYLINE / INSIGHTS")
st.markdown("""
- How do users behave during pre and live matches?
- Which sport event has the higher likelihood of profitability?
- Age-grade betting with multi-betting preference and its correlation. Which age group/bracket are risk-takers or risk-averse?
- Understanding the risk and reward associated with multi-bets involving different combinations of sports/Correlation between age and multi-bet.
- First bet users demographic and how it relates to age.
- What insight can help to increase betting among betting users?
- What sport event do users tend to spend more despite the odds?
""")

# Load the dataset
st.markdown("## Data Loading and Overview")
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel('HomeworDATA.xlsx')

    # Display the dataframe
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Display Data Info
    buffer = st.empty()  # Placeholder for text
    df_info = buffer.text("Data Info:\n" + str(df.info()))

    # Display Dataframe Head
    st.write("### Data Head")
    st.write(df.head())

    # Check and create necessary columns
    if 'Odds - Event' not in df.columns:
        df['Odds - Event'] = df['Bet Odds']

    if 'Bet Amount per Event' not in df.columns:
        df['Bet Amount per Event'] = df['Bet Amount']

    # Ensure 'Is Multi Bet' is a boolean
    df['Is Multi Bet'] = df['Is Multi Bet'].astype(bool)

    # Visualizations
    st.markdown("## Data Visualizations")

    # User Age Distribution
    st.write("### User Age Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['User Age'], bins=20, kde=True)
    st.pyplot()

    # Bet Amount by Event Type
    st.write("### Bet Amount by Event Type")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Event Type', y='Bet Amount', data=df)
    st.pyplot()

    # Pairplot
    st.write("### Pairplot of Selected Columns")
    sns.pairplot(df[['User Age', 'Bet Amount', 'Bet Odds', 'Odds - Event']])
    st.pyplot()

    # Violin Plot by Sport and Bet Amount
    st.write("### Violin Plot: Bet Amount by Sport")
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Sport', y='Bet Amount', data=df, scale='width')
    st.pyplot()

    # KMeans Clustering
    st.markdown("## KMeans Clustering Analysis")
    
    # Scaling features for clustering
    scaler = StandardScaler()
    features = df[['User Age', 'Bet Amount', 'Bet Odds']]
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    # Display cluster centers
    st.write("### Cluster Centers")
    st.write(kmeans.cluster_centers_)

    # Display Data with Clusters
    st.write("### Data with Cluster Labels")
    st.dataframe(df.head())

    # Scatter plot for clusters
    st.write("### Clusters Visualization")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='User Age', y='Bet Amount', hue='Cluster', data=df, palette='viridis')
    st.pyplot()

    # Profitability by Event Type
    st.write("### Profitability by Event Type")
    profitability = df.groupby('Event Type')['Bet Amount'].sum()
    st.bar_chart(profitability)

    # Correlation Matrix
    st.write("### Correlation Matrix")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot()

    # Insights on Multi Bets
    st.markdown("### Insights on Multi Bets")
    multi_bet_insights = df[df['Is Multi Bet'] == True]
    st.write("Number of Multi Bets: ", len(multi_bet_insights))

    st.write("Multi Bet Distribution by Sport")
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Sport', data=multi_bet_insights)
    st.pyplot()

    # Multi Bet Amount by User Age
    st.write("Multi Bet Amount by User Age")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='User Age', y='Bet Amount', data=multi_bet_insights)
    st.pyplot()

    # Sports Correlation with Age Group
    st.write("### Sports Correlation with Age Group")
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Sport', y='User Age', data=df, scale='width')
    st.pyplot()

    # Insights and Conclusion
    st.markdown("## Conclusion and Insights")
    st.markdown("""
    This analysis provides a deep understanding of user behavior during pre and live matches. By analyzing the risk and reward
    associated with multi-bets and understanding demographic patterns, this information can help improve user engagement,
    target marketing efforts, and enhance profitability.
    """)

