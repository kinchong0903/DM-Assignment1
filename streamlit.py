import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="TDS 3301 - Assignment 1",
    # layout="wide",
    initial_sidebar_state="expanded",
)

st.write("""
# TDS 3301 - Assignment 1

Chong Wing Kin - 1191302246

Koh Kai Sheng - 1151104252

Lee Jun Yong - 1191302186

# (i) Exploratory Analysis
## - Cases by state
""")

df_cases_state = pd.read_csv("cases_state.csv")

st.dataframe(df_cases_state)

st.write("""
This dataset gives the information of daily recorded COVID-19 cases at state level.
Each state has one row every day that provides the information above.

This dataset is available from 2020-01-25 to 2021-09-14

Preliminary analysis shows that there are no missing values for all columns.

""")

st.dataframe(df_cases_state.groupby('state').count())

st.write("""
Dataframe above shows the dataset grouped by state. This was done to check if there are any noise with the labels of state.

From the dataframe above, it shows no repetition of states which means the labels of state are identical for each state in the rows.
It also shows that every states have the same amount of data (599 rows).

## - Tests by state
This dataset is the daily tests (note: not necessarily unique individuals) by type at state level.
""")

df_tests_state = pd.read_csv("tests_state.csv") 
st.dataframe(df_tests_state)

st.write("""
This dataset is only available from 2021-07-01 to 2021-09-12 which is significantly less than the
dataset available for cases. Due to this reason, our team have decided to not use this dataset in our project.


## - Clusters
""")
df_clusters = pd.read_csv("clusters.csv")
st.dataframe(df_clusters)

st.write("""
Dataframe above shows the clusters that were announced by KKM. 

From preliminary analysis of the dataset, we have found out that the states that are related to the clusters
are listed in the 'state' column, separated by commas if the cluster involves multiple state. This is a different way of showing state than what was with
the other datasets we have in this project. Due to this reason, further transformation will be done to this dataset in order for us to use this dataset.

We have also found out that there are no missing values in all columns from this dataset.

Other than that, we have found out that this dataset contains data from 2020-03-01 to 2021-09-13. Although the duration of data is slightly less than the cases dataset,
this data will be sufficient for our use so we will be using this dataset in our project.

## - ICU by state
""")

df_icu = pd.read_csv("icu.csv")
st.dataframe(df_icu)
st.write("""
Dataframe above shows the capacity and utilisation of intensive care unit (ICU) beds.

In this project we have decided to use the icu_covid column from the dataset which is the number of ICU being used by patients by each state each date.
This dataset contains data from 2020-03-24 to 2021-09-18. Similar to the clusters dataset, slightly less than the cases dataset, this data will be sufficient for our use so we will be using this dataset in our project.

## - Cleaned and Transformed Dataframe
""")

df = pd.read_csv("generatedCsv/new_df.csv")
st.dataframe(df)
st.write("""
Dataframe above shows the final dataframe that has been cleaned and transformed.
Clusters have been combined to the cases table to each row which gives the total clusters announced each day. Initial plan was to have the active clusters each day but this data was not provided by the KKM.
We have also merged the column 'icu_covid' to the dataframe to show the utilisation of ICU each day by each state.

Two columns each have also been added for the column cases_new, total_clusters_announced and icu_covid. These 2 columns are the data of each of the 3 mentioned column but made to be the 7 and 14 days moving average data.
These features will be used in the later stages of this project building models and making predictions.

From the mentioned moving average data, we have made a column each which is offset by last 14 days from the data. This so that we can use it in our model and if done successfully, the model can be used to predict cases 14 days later given the current day data.

## - Plots
Below are some plots we have made to showcase a clearer picture on our datasets.
""")
img = Image.open("images/new_cases_per_mth.png")
st.image(img)
img = Image.open("images/total_clusters_announced_per_mth.png")
st.image(img)
img = Image.open("images/avg_icu_per_mth.png")
st.image(img)

st.write("""
# (ii) Correlation (Pahang & Johor)
""")

img = Image.open("images/correlation_matrix_cases_new.png")
st.image(img)

st.write("""
Figure above shows the correlation matrix of daily cases between each states. From the correlation matrix we are able to find states that have strong correlation with Pahang and Johor.

## Pahang
States that have >0.9 correlation value with Pahang
""")
st.markdown(
    """
| state | correlation_value |
| --- | --- |
| Kedah | 0.942919 |
| Perak | 0.907643 |
| Terengganu | 0.913044 |


## Johor
States that have >0.9 correlation value with Johor

| state | correlation_value |
| --- | --- |
| Kedah | 0.900592 |
| Perak | 0.921998 |
| Pulau Pinang | 0.926004 |
| Terengganu | 0.913753 |
"""
)
st.write("""

# (iii) Features/Indicators
In this study we have adopted two methods to select the features for training models in our next stage. For each of the states, we have used SelectKBest and Recursive Feature Elimination (RFE).

## Pahang
### SelectKBest
Selected Feature: cases_new_mva_7_days_offset_14_days
""")
img = Image.open("images/SelectKBestPahang.png")
st.image(img)
st.write("""
### Recursive Feature Elimination (RFE)
Selected Features : 
- cases_new_mva_7_days_offset_14_days
- clusters_announced_mva_7_days_offset_14_days
- clusters_announced_mva_14_days_offset_14_days
- icu_covid_mva_7_days_offset_14_days
---
## Kedah 
### SelectKBest
Selected Feature: icu_covid_mva_7_days_offset_14_days
""")
img = Image.open("images/SelectKBestKedah.png")
st.image(img)

st.write("""
### Recursive Feature Elimination (RFE)
Selected Features : 
- clusters_announced_mva_7_days_offset_14_days
- clusters_announced_mva_14_days_offset_14_days
- icu_covid_mva_7_days_offset_14_days
- icu_covid_mva_14_days_offset_14_days
---
## Johor 
### SelectKBest
Selected Feature: cases_new_mva_7_days_offset_14_days
""")
img = Image.open("images/SelectKBestJohor.png")
st.image(img)
st.write("""
### Recursive Feature Elimination (RFE)
- cases_new_mva_7_days_offset_14_days
- clusters_announced_mva_7_days_offset_14_days
- clusters_announced_mva_14_days_offset_14_days
- icu_covid_mva_7_days_offset_14_days
---
## Selangor
### SelectKBest
Selected Feature: icu_covid_mva_14_days_offset_14_days
""")
img = Image.open("images/SelectKBestSelangor.png")
st.image(img)
st.markdown("""
### Recursive Feature Elimination (RFE)
- clusters_announced_mva_7_days_offset_14_days
- clusters_announced_mva_14_days_offset_14_days
- icu_covid_mva_7_days_offset_14_days
- icu_covid_mva_14_days_offset_14_days
""")

# Regression and classification Models
st.markdown("""
# (iv) Regression & Classification Models
## Pahang
### Regression
Linear Regression 
- Score: 0.7882766418397797 
- MSE: 6517.32666582556 
""")
img = Image.open("images/linear_regression_pahang.png")
st.image(img)
st.markdown("""
Ridge Regression
- Score : 0.795032194884211
- MSE: 6309.375373245205
""")
img = Image.open("images/new_cases_prediction_pahang.png")
st.image(img)
st.markdown("""
### Classification

- KNN score:  0.7538461538461538
- Decision Tree score:  0.7615384615384615
""")
img = Image.open("images/PahangDT.png")
st.image(img)

# Kedah
st.markdown("""
---
## Kedah
### Regression
Linear Regression 
- Score: 0.7768646572691711 
- MSE: 42695.89877947014
""")
img = Image.open("images/linear_regression_kedah.png")
st.image(img)
st.markdown("""
Ridge Regression
- Score : 0.7698503858459211
- MSE: 44038.04663930102
""")
img = Image.open("images/new_cases_prediction_kedah.png")
st.image(img)
st.markdown("""
### Classification

- KNN score:  0.797979797979798
- Decision Tree score:  0.7626262626262627
""")
img = Image.open("images/KedahDT.png")
st.image(img)

# Johor
st.markdown("""
---
## Johor
### Regression
Linear Regression 
- Score: 0.8143979284231753 
- MSE: 44403.324678928715 
""")
img = Image.open("images/linear_regression_johor.png")
st.image(img)
st.markdown("""
Ridge Regression
- Score : 0.7366375388444197
- MSE: 63006.672132387845
""")
img = Image.open("images/new_cases_prediction_johor.png")
st.image(img)
st.markdown("""
### Classification

- KNN score:  0.7288135593220338
- Decision Tree score:  0.7796610169491526
""")
img = Image.open("images/JohorDT.png")
st.image(img)
st.markdown("""
---
## Selangor
### Regression
Linear Regression 
- Score: 0.7435731383578753 
- MSE: 654674.2196266949
""")
img = Image.open("images/linear_regression_selangor.png")
st.image(img)
st.markdown("""
Ridge Regression
- Score : 0.7499459997018736
- MSE: 638403.8959934637
""")
img = Image.open("images/new_cases_prediction_selangor.png")
st.image(img)
st.markdown("""
### Classification

- KNN score:  0.89501312335958
- Decision Tree score:  0.863517060367454
""")
img = Image.open("images/SelangorDT.png")
st.image(img)
st.markdown("""
---
""")