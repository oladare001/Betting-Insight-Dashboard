#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Betting Analysis Dashboard", layout="wide")

# Title and introduction
st.title("Betting Analysis Dashboard")
st.markdown("""
This dashboard provides insights into betting behavior, including user demographics, 
betting patterns, and profitability analysis.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('HomeworDATA.xlsx')
    
    # Check if 'Odds - Event' and 'Bet Amount per Event' columns exist
    if 'Odds - Event' not in df.columns:
        df['Odds - Event'] = df['Bet Odds']
    
    if 'Bet Amount per Event' not in df.columns:
        df['Bet Amount per Event'] = df['Bet Amount']
    
    # Convert 'Is Multi Bet' to boolean if it's not already
    df['Is Multi Bet'] = df['Is Multi Bet'].astype(bool)
    
    return df

df = load_data()

# Data Distribution
st.header("Data Distribution")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df['User Age'], bins=5, kde=True, ax=ax)
    ax.set_title('Age Distribution of Bettors')
    ax.set_xlabel('Age')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    df['User Language'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title('Language Distribution')
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots()
    df['Age Group'] = pd.cut(df['User Age'], bins=[18, 25, 35, 45, 55, 65, 80], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    sns.barplot(x='Age Group', y='Bet Amount per Event', data=df, ax=ax)
    ax.set_title('Average Bet Amount per Event by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Average Bet Amount per Event')
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots()
    df['Bet Type'] = np.where(df['Is Multi Bet'], 'Multi', 'Single')
    df['Bet Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title('Bet Type Preference')
    st.pyplot(fig)

st.markdown(f"""
- Average bettor age: {df['User Age'].mean():.2f}
- Most common language: {df['User Language'].mode()[0]}
- Average bet amount per event: ${df['Bet Amount per Event'].mean():.2f}
""")

# Average Amount spent Event / Odd -Event Distribution
st.header("Average Amount spent Event / Odd -Event Distribution")

col5, col6 = st.columns(2)

with col5:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Sport', y='Bet Amount per Event', data=df, ax=ax)
    ax.set_title('Average Bet Amount per Event by Sport')
    ax.set_xlabel('Sport')
    ax.set_ylabel('Average Bet Amount per Event')
    plt.xticks(rotation=90)
    st.pyplot(fig)

with col6:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Sport', y='Odds - Event', data=df, ax=ax)
    ax.set_title('Odds Distribution by Sport')
    ax.set_xlabel('Sport')
    ax.set_ylabel('Odds')
    plt.xticks(rotation=90)
    st.pyplot(fig)

#st.dataframe(df.groupby('Sport')[['Bet Amount per Event', 'Odds - Event']].agg(['mean', 'median']))


# Perform the groupby operation
sport_stats = df.groupby('Sport').agg({
    'Bet Amount per Event': ['mean', 'median'],
    'Odds - Event': ['mean', 'median']
})

# Flatten the column names
sport_stats.columns = ['_'.join(col).strip() for col in sport_stats.columns.values]

# Reset the index to make 'Sport' a column
sport_stats = sport_stats.reset_index()

# Custom formatting function
def format_float(val):
    return f"{val:.6f}".rstrip('0').rstrip('.')

# Apply custom formatting
formatted_sport_stats = sport_stats.copy()
for col in sport_stats.columns[1:]:  # Skip the 'Sport' column
    formatted_sport_stats[col] = sport_stats[col].apply(format_float)

# Display the formatted dataframe
st.dataframe(formatted_sport_stats)




# Amount per Event Distribution
st.header("Amount per Event Distribution")
st.markdown("The spread of how people spend per event")

fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(df['Bet Amount per Event'], kde=True, bins=10, ax=ax)
ax.set_title('Distribution of Bet Amounts per Event')
ax.set_xlabel('Bet Amount per Event')
ax.set_ylabel('Frequency')
ax.axvline(df['Bet Amount per Event'].median(), color='r', linestyle='--', label='Median')
ax.axvline(df['Bet Amount per Event'].mean(), color='g', linestyle='--', label='Mean')
ax.legend()
st.pyplot(fig)

st.markdown(f"""
- Mean Bet Amount: ${df['Bet Amount per Event'].mean():.2f}
- Median Bet Amount: ${df['Bet Amount per Event'].median():.2f}
- Standard Deviation: ${df['Bet Amount per Event'].std():.2f}
""")

# Event type Distribution
st.header("Event Type Distribution")

col7, col8 = st.columns(2)

with col7:
    fig, ax = plt.subplots()
    event_type_dist = df['Event Type'].value_counts(normalize=True) * 100
    event_type_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title('Distribution of Pre-match vs. In-play Bets')
    st.pyplot(fig)

with col8:
    fig, ax = plt.subplots()
    event_type_by_sport = df.groupby('Sport')['Event Type'].value_counts(normalize=True).unstack()
    event_type_by_sport.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Event Type Distribution by Sport')
    ax.set_xlabel('Sport')
    ax.set_ylabel('Percentage of Bets by Event Type')
    plt.legend(title='Event Type')
    plt.xticks(rotation=90)
    st.pyplot(fig)

st.dataframe(event_type_by_sport)

avg_bet_by_event_type = df.groupby('Event Type')['Bet Amount per Event'].mean()
st.markdown("Average Bet Amount per Event:")
st.dataframe(avg_bet_by_event_type)

# Multi betting Preference
st.header("Multi Betting Preference")
st.markdown("The Dynamics between Age and Multi betting.")

fig, ax = plt.subplots(figsize=(10, 6))
age_bet_type = df.groupby('User Age')['Is Multi Bet'].mean()
ax.plot(age_bet_type.index, age_bet_type.values * 100)
ax.set_xlabel('User Age')
ax.set_ylabel('Percentage of Multi-Bets')
ax.set_title('Correlation between User Age and Multi-Bet Preference')
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

correlation = df['User Age'].corr(df['Is Multi Bet'].astype(int))
st.markdown(f"Correlation between age and multi-bet preference: {correlation:.2f}")

age_groups = pd.cut(df['User Age'], bins=[18, 25, 35, 45, 55, 65, 80], labels=['18-25', '26-35', '36-45',
                                                                               '46-55', '56-65', '65+'],include_lowest=True)
multi_bet_by_age_group = df.groupby(age_groups)['Is Multi Bet'].mean() * 100


multi_bet_by_age_group_df = multi_bet_by_age_group.reset_index()
multi_bet_by_age_group_df.columns = ['Age Group', 'Percentage of Multi-Bets']

st.markdown("Percentage of Multi-Bets by Age Group:")
st.dataframe(multi_bet_by_age_group_df.style.format({'Percentage of Multi-Bets': '{:.2f}%'}))

avg_bet_amount = df.groupby('Is Multi Bet')['Bet Amount per Event'].mean()
st.markdown("Average Bet Amount per Event:")
st.markdown(f"Multi-Bets: ${avg_bet_amount[True]:.2f}")
st.markdown(f"Single Bets: ${avg_bet_amount[False]:.2f}")

# First Bet Analysis by Age Group
st.header("First Bet Analysis by Age Group")
st.markdown("""
No strong relation exists between first-time betters and amount spent, but as age increases, 
the size of amount spent increases among bet users.

An incentive for first-timers or those below a certain age can be offered for them to spend more on betting.
""")

first_bets = df[df['Is First Bet'] == 'Y']
age_groups = pd.cut(first_bets['User Age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                    labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], include_lowest=True)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(first_bets['User Age'], first_bets['Bet Amount per Event'], alpha=0.5)
ax.set_xlabel('User Age')
ax.set_ylabel('First Bet Amount')
ax.set_title('First Bet Amounts by User Age')
ax.grid(True, linestyle='--', alpha=0.7)

z = np.polyfit(first_bets['User Age'], first_bets['Bet Amount per Event'], 1)
p = np.poly1d(z)
ax.plot(first_bets['User Age'], p(first_bets['User Age']), "r--", alpha=0.8)
st.pyplot(fig)

avg_first_bet = first_bets['Bet Amount per Event'].mean()
median_first_bet = first_bets['Bet Amount per Event'].median()
correlation_age_amount = first_bets['User Age'].corr(first_bets['Bet Amount per Event'])

st.markdown("First Bet Statistics:")
st.markdown(f"- Average first bet amount: ${avg_first_bet:.2f}")
st.markdown(f"- Median first bet amount: ${median_first_bet:.2f}")
st.markdown(f"- Correlation between age and first bet amount: {correlation_age_amount:.2f}")

first_bet_by_age_group = first_bets.groupby(age_groups)['Bet Amount per Event'].agg(['mean', 'median', 'count'])
first_bet_by_age_group.columns = ['Average Amount', 'Median Amount', 'Number of First Bets']
first_bet_by_age_group = first_bet_by_age_group.reset_index()
first_bet_by_age_group.columns = ['Age Group'] + list(first_bet_by_age_group.columns[1:])

st.markdown("First Bet Analysis by Age Group:")
st.dataframe(first_bet_by_age_group.style.format({
    'Average Amount': '${:.2f}',
    'Median Amount': '${:.2f}',
    'Number of First Bets': '{:.0f}'
}))
first_bet_percentage = (df['Is First Bet'] == 'Y').mean() * 100
st.markdown(f"Percentage of bets that are first-time bets: {first_bet_percentage:.2f}%")

# User Behaviour with Multi bet and Event odds
st.header("User Behaviour with Multi bet and Event odds")
st.markdown("""
To what extent or size of event do bet users commit betting in a Multi bet scenario.

From this image, The number of sport events users take on high odds is 2 types of sport events.
The higher the number of sport event combinations in a multi-bet series, the lower the odds.

Understanding how overall odds affect the diversity of sports in a multi-bet can be valuable 
for identifying patterns in betting behavior and potentially optimizing strategies for multi-bets.
""")

multi_bets = df[df['Is Multi Bet']]
multi_bet_odds = multi_bets.groupby('Bet ID')['Odds - Event'].prod().reset_index()
multi_bet_odds.columns = ['Bet ID', 'Total Odds']
multi_bet_analysis = multi_bets.merge(multi_bet_odds, on='Bet ID')

fig, ax = plt.subplots(figsize=(12, 6))
sports_count = multi_bets.groupby('Bet ID')['Sport'].nunique()
ax.scatter(sports_count, multi_bet_odds['Total Odds'])
ax.set_xlabel('Number of Different Sports in Multi-Bet')
ax.set_ylabel('Total Odds')
ax.set_title('Impact of Sport Combination on Multi-Bet Odds')
st.pyplot(fig)

st.markdown(f"Average total odds for multi-bets: {multi_bet_odds['Total Odds'].mean():.2f}")
st.markdown(f"Average number of sports in multi-bets: {sports_count.mean():.2f}")

# Clustering with Age
st.header("Clustering with Age")
st.markdown("""
Analysis reveals that age clustering is indeed an important factor in differentiating
betting behaviors, with distinct patterns emerging for different age groups in
terms of bet sizes, odds, and propensity for multi-betting.
""")

features = ['User Age', 'Bet Amount per Event', 'Odds - Event', 'Is Multi Bet']
X = df[features].copy()
X['Is Multi Bet'] = X['Is Multi Bet'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_analysis = df.groupby('Cluster')[features].mean()
cluster_analysis['Size'] = df['Cluster'].value_counts()
cluster_analysis = cluster_analysis.sort_values('Size', ascending=False)

st.markdown("Betting Behavior Segments:")
st.dataframe(cluster_analysis)

fig, ax = plt.subplots(figsize=(12, 8))
for cluster in sorted(df['Cluster'].unique()):
    subset = df[df['Cluster'] == cluster]
    ax.scatter(subset['User Age'], subset['Bet Amount per Event'], 
               label=f'Cluster {cluster}', alpha=0.6)
ax.set_xlabel('User Age')
ax.set_ylabel('Bet Amount per Event')
ax.set_title('Betting Behavior Segments')
ax.legend(title='Cluster')
st.pyplot(fig)

# Profitability Insight
st.header("Profitability Insight")
st.markdown("""
Understanding Users' behaviour by The popularity of each Sport event among users, average amount spent on each sport event
and using the odd event to understand potential profitability will be insightful for bet owners in setting bet odds.
""")

sport_analysis = df.groupby('Sport').agg({
    'Bet ID': 'count',
    'Bet Amount per Event': 'mean',
    'Odds - Event': 'mean'
}).sort_values('Bet ID', ascending=False)

sport_analysis.columns = ['Number of Bets', 'Avg Bet Amount', 'Avg Odds']
sport_analysis['Popularity Rank'] = sport_analysis['Number of Bets'].rank(ascending=False)
sport_analysis['Potential Profitability'] = sport_analysis['Avg Bet Amount'] * (sport_analysis['Avg Odds'] - 1)

fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=sport_analysis.reset_index(), x='Popularity Rank', y='Potential Profitability', 
                size='Avg Bet Amount', hue='Sport', sizes=(20, 200), ax=ax)
ax.set_title('Sport Popularity vs Potential Profitability')
ax.legend(title='Sport', loc='upper right', bbox_to_anchor=(1.15, 1))
ax.set_xlabel('Popularity Rank (1 = Most Popular)')
ax.set_ylabel('Potential Profitability')
st.pyplot(fig)

st.dataframe(sport_analysis)

# Comprehensive analysis of user betting behavior
st.header("Comprehensive analysis of user betting behavior")
st.markdown("""
This analysis provides a comprehensive view of how age, betting frequency, bet amounts, 
and multi-bet preferences interact in your user base, helping identify patterns and 
potential user segments for targeted strategies.

High-Value User Statistics also show the big spenders or those in the top 10% (90th percentile).
""")

user_analysis = df.groupby('User ID').agg({
    'Bet ID': 'count',
    'Bet Amount per Event': 'mean',
    'Is Multi Bet': 'mean',
    'Odds - Event': 'mean',
    'User Age': 'first'
})

user_analysis.columns = ['Number of Bets', 'Avg Bet Amount', 'Avg Multi Bet', 'Avg Odds', 'Age']

fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(user_analysis['Age'], user_analysis['Avg Bet Amount'], 
                     c=user_analysis['Avg Multi Bet'], s=user_analysis['Number of Bets'],
                     alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Avg Multi Bet')
ax.set_title('User Betting Behavior')
ax.set_xlabel('User Age')
ax.set_ylabel('Average Bet Amount')
st.pyplot(fig)

correlation_matrix = user_analysis.corr()
st.subheader("Correlation Matrix")
st.dataframe(correlation_matrix)

high_value_users = user_analysis[user_analysis['Avg Bet Amount'] > user_analysis['Avg Bet Amount'].quantile(0.9)]
st.subheader("High-Value User Statistics")
st.dataframe(high_value_users.describe())

# Add a sidebar for navigation
st.sidebar.title("Navigation")
pages = [
    "Data Distribution",
    "Average Amount and Odds Distribution",
    "Amount per Event Distribution",
    "Event Type Distribution",
    "Multi Betting Preference",
    "First Bet Analysis",
    "Multi-bet and Event Odds",
    "Clustering with Age",
    "Profitability Insight",
    "User Betting Behavior"
]
selection = st.sidebar.radio("Go to", pages)

# You can add logic here to show/hide sections based on the selection
# For example:
# if selection == "Data Distribution":
#     st.header("Data Distribution")
#     # Show only the Data Distribution section
# elif selection == "Average Amount and Odds Distribution":
#     st.header("Average Amount and Odds Distribution")
#     # Show only the Average Amount and Odds Distribution section
# ... and so on for each section

# Add some final notes or conclusions
st.markdown("""
## Conclusions and Recommendations

Based on the analysis, here are some key takeaways and potential recommendations:

1. Age plays a significant role in betting behavior, with different age groups showing
distinct patterns in bet amounts and multi-bet preferences.
2. There's potential to encourage first-time bettors, especially younger users, with targeted incentives.
3. Multi-betting behavior varies with age and could be a focus for marketing strategies.
4. Certain sports show higher potential profitability and could be promoted more heavily.
5. High-value users have distinct characteristics and could be targeted for retention strategies.
""")

