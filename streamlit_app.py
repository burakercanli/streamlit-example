from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from typing import DefaultDict
from pandas._libs.tslibs.timedeltas import Timedelta
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

"""
# Welcome to Employee Management System!

The dashboard can show you detailed information for entire factory or a specific employee.
"""

df = pd.read_csv("hackathon.csv", parse_dates=["date"],sep="\t")
df = df.sort_values("date")

st.write(df.head())


elist = df['name'].unique()
employee = st.sidebar.selectbox("Select a employee:",elist)

# Toplam Çalışma Süresi
total_work = Timedelta(0)
worker_dict = {}
worker_time_dict = {}
for i, row in df.iterrows():
  if row["entrance"] == 1:
    worker_dict[row["name"]] = row.date
  else:
    temp_dur = row.date - worker_dict[row["name"]]
    worker_time_dict[row["name"]] = worker_time_dict.get(row["name"], Timedelta(0)) + temp_dur
    total_work += temp_dur
print(total_work)

# Kişi çalışma Süresi
print(worker_time_dict)

# Toplam mola süresi
# Çalışan başına mola süresi
total_break = Timedelta(0)
worker_dict = {}
worker_break_dict = {}
num_breaks_worker = {}
for i, row in df.iterrows():
  if row["entrance"] == 0 or row["name"] not in worker_dict:
    worker_dict[row["name"]] = row.date
  else:
    temp_dur = row.date - worker_dict[row["name"]]
    if temp_dur < Timedelta("03:00:00"):
      num_breaks_worker[row["name"]] = num_breaks_worker.get(row["name"], 0) + 1
      worker_break_dict[row["name"]] = worker_break_dict.get(row["name"], Timedelta(0)) + temp_dur
      total_break += temp_dur
print(total_break)
print(worker_break_dict)

# Ortalama mola sayısı
print(num_breaks_worker)

# Ortalama mola süresi
avg_break_worker = {}
for worker in worker_break_dict:
  avg_break_worker[worker] = worker_break_dict[worker] / num_breaks_worker[worker]
print(avg_break_worker)

# Çalışan duygu durumu
worker_emotion_dict = {}
for worker in worker_time_dict:
  temp_dict = {}
  temp_dict["positive"] = df[df["name"] == worker].positive.sum()
  temp_dict["negative"] = df[df["name"] == worker].negative.sum()
  temp_dict["neutral"] = df[df["name"] == worker].neutral.sum()
  max_val = max(temp_dict["positive"], temp_dict["negative"], temp_dict["neutral"])
  if max_val == temp_dict["positive"]:
    temp_dict["sentiment"] = "positive"
  elif max_val == temp_dict["negative"]:
    temp_dict["sentiment"] = "negative"
  else:
    temp_dict["sentiment"] = "neutral"
  worker_emotion_dict[worker] = temp_dict
print(worker_emotion_dict)

# Genel duygu durumu
overall_sentiment = {}
for worker in worker_emotion_dict:
  overall_sentiment["positive"] = overall_sentiment.get("positive", 0) + worker_emotion_dict[worker]["positive"]
  overall_sentiment["negative"] = overall_sentiment.get("negative", 0) + worker_emotion_dict[worker]["negative"]
  overall_sentiment["neutral"] = overall_sentiment.get("neutral", 0) + worker_emotion_dict[worker]["neutral"]

max_val = max(overall_sentiment["positive"], overall_sentiment["negative"], overall_sentiment["neutral"])
if max_val == overall_sentiment["positive"]:
  overall_sentiment["sentiment"] = "positive"
elif max_val == overall_sentiment["negative"]:
  overall_sentiment["sentiment"] = "negative"
else:
  overall_sentiment["sentiment"] = "neutral"
  
print(overall_sentiment)

st.markdown("## Entire Factory")

kpi1, kpi2 = st.columns(2)

with kpi1:
  st.markdown("**Total Work Hours**")
  st.markdown(f"<h1 style='text-align: center; color: red;'>{total_work}</h1>", unsafe_allow_html=True)
  
with kpi2:
    st.markdown("**Total Break Hours**")
    st.markdown(f"<h1 style='text-align: center; color: red;'>{total_break}</h1>", unsafe_allow_html=True)

st.markdown("<hr/>",unsafe_allow_html=True)

st.title("General Sentiment")
fig = px.pie(overall_sentiment, values=overall_sentiment.values()[:3], names=overall_sentiment.keys()[:3], title='Total Sentiment Cases')
st.plotly_chart(fig)


#st.markdown("## General Sentiment")

#kpi01, kpi02, kpi03, kpi04 = st.columns(4)
    
#with kpi01:
 #   st.markdown("**General Sentiment**")
  #  st.markdown(f"<h1 style='text-align: center; color: red;'>{overall_sentiment}</h1>", unsafe_allow_html=True)


#if page == 'burak':
  # Display burak
#elif page == 'taylan':
  # Display taylan
#else :
    #display berhan

