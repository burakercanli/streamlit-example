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

def calculate_wellness_productivity(name):
  prod = (worker_emotion_dict[name]['positive'] * worker_time_dict[name] * 0.06 - worker_emotion_dict[name]['negative'] * worker_break_dict[name]).__str__().split()[0]
  prod = int(prod)

  well = (worker_emotion_dict[name]['positive'] * worker_break_dict[name] - worker_emotion_dict[name]['negative'] * worker_time_dict[name] * 0.06).__str__().split()[0]
  well = int(well)

  return prod, well

df = pd.read_csv("hackathon.csv", parse_dates=["date"],sep="\t")
df = df.sort_values("date")
df["neutral"] = 100 - (df.positive + df.negative)

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

overall_sentiment["positive"] = [overall_sentiment.get("positive", 0)]
overall_sentiment["negative"] = [overall_sentiment.get("negative", 0)]
overall_sentiment["neutral"] = [overall_sentiment.get("neutral", 0)]

max_val = max(overall_sentiment["positive"], overall_sentiment["negative"], overall_sentiment["neutral"])
if max_val == overall_sentiment["positive"]:
  overall_sentiment["sentiment"] = ["positive"]
elif max_val == overall_sentiment["negative"]:
  overall_sentiment["sentiment"] = ["negative"]
else:
  overall_sentiment["sentiment"] = ["neutral"]
  
print(overall_sentiment)
overall_sentiment = pd.DataFrame.from_dict(overall_sentiment)

st.markdown("## Entire Factory")

kpi1, kpi2 = st.columns(2)

with kpi1:
  st.markdown("**Total Work Hours**")
  st.markdown(f"<h1 style='text-align: center; color: green;'>{total_work}</h1>", unsafe_allow_html=True)
  
with kpi2:
    st.markdown("**Total Break Hours**")
    st.markdown(f"<h1 style='text-align: center; color: green;'>{total_break}</h1>", unsafe_allow_html=True)

st.markdown("<hr/>",unsafe_allow_html=True)

st.title("General Sentiment")
fig = px.pie(overall_sentiment, values=overall_sentiment.iloc[0, :3], names=overall_sentiment.columns[:3], title='Total Sentiment Cases')
st.plotly_chart(fig)


st.markdown("## Sentiment Result")
st.markdown(f"<h1 style='text-align: center; color: yellow;'>{overall_sentiment.loc[0, 'sentiment'].title()}</h1>", unsafe_allow_html=True)


elist = ['None'] + list(df['name'].unique())
employee = st.selectbox("Select a employee:", elist)

if employee == 'burak':
  st.markdown("## Statistics for Burak")

  kpi3, kpi4, kpi5 = st.columns(3)

  with kpi3:
    st.markdown("**Total Work Hours**")
    st.markdown(f"<h1 style='text-align: center; color: yellow;'>{worker_time_dict.get('burak')}</h1>", unsafe_allow_html=True)

  with kpi4:
      st.markdown("**Total Break Hours**")
      st.markdown(f"<h1 style='text-align: center; color: green;'>{worker_break_dict.get('burak')}</h1>", unsafe_allow_html=True)
      
  with kpi5:
      st.markdown("**Total Number of Breaks**")
      st.markdown(f"<h1 style='text-align: center; color: green;'>{num_breaks_worker.get('burak')}</h1>", unsafe_allow_html=True)

  st.markdown("<hr/>",unsafe_allow_html=True)
  
  kpi6, kpi7 = st.columns(2)
  
  well, prod = calculate_wellness_productivity('burak')
  
  with kpi6:
      st.markdown("**Wellness Index**")
      st.markdown(f"<h1 style='text-align: center; color: yellow;'>{well//10}</h1>", unsafe_allow_html=True)
      
  with kpi7:
      st.markdown("**Productivity Index**")
      st.markdown(f"<h1 style='text-align: center; color: yellow;'>{prod//10}</h1>", unsafe_allow_html=True)
  
  st.markdown("**Recent Emotional State**")
  st.line_chart(df[df['name'] == 'burak'][['positive', 'negative']])
elif employee == 'taylan':
  st.markdown("## Statistics for Taylan")

  kpi3, kpi4, kpi5 = st.columns(3)

  with kpi3:
    st.markdown("**Total Work Hours**")
    st.markdown(f"<h1 style='text-align: center; color: green;'>{worker_time_dict.get('taylan')}</h1>", unsafe_allow_html=True)

  with kpi4:
      st.markdown("**Total Break Hours**")
      st.markdown(f"<h1 style='text-align: center; color: yellow;'>{worker_break_dict.get('taylan')}</h1>", unsafe_allow_html=True)
      
  with kpi5:
      st.markdown("**Total Number of Breaks**")
      st.markdown(f"<h1 style='text-align: center; color: green;'>{num_breaks_worker.get('taylan')}</h1>", unsafe_allow_html=True)

  st.markdown("<hr/>",unsafe_allow_html=True)
  
  kpi6, kpi7 = st.columns(2)
  well, prod = calculate_wellness_productivity('taylan')
  with kpi6:
      st.markdown("**Wellness Index**")
      st.markdown(f"<h1 style='text-align: center; color: yellow;'>{well//10}</h1>", unsafe_allow_html=True)
      
  with kpi7:
      st.markdown("**Productivity Index**")
      st.markdown(f"<h1 style='text-align: center; color: green;'>{prod//10}</h1>", unsafe_allow_html=True)
      
  # Display taylan
  st.markdown("**Recent Emotional State**")
  st.line_chart(df[df['name'] == 'taylan'][['positive', 'negative']])
elif employee == 'berhan':
  st.markdown("## Statistics for Berhan")

  kpi3, kpi4, kpi5 = st.columns(3)

  with kpi3:
    st.markdown("**Total Work Hours**")
    st.markdown(f"<h1 style='text-align: center; color: green;'>{worker_time_dict.get('berhan')}</h1>", unsafe_allow_html=True)

  with kpi4:
      st.markdown("**Total Break Hours**")
      st.markdown(f"<h1 style='text-align: center; color: green;'>{worker_break_dict.get('berhan')}</h1>", unsafe_allow_html=True)
      
  with kpi5:
      st.markdown("**Total Number of Breaks**")
      st.markdown(f"<h1 style='text-align: center; color: green;'>{num_breaks_worker.get('berhan')}</h1>", unsafe_allow_html=True)
  
  st.markdown("<hr/>",unsafe_allow_html=True)
  
  kpi6, kpi7 = st.columns(2)
  well, prod = calculate_wellness_productivity('berhan')
  
  with kpi6:
      st.markdown("**Wellness Index**")
      st.markdown(f"<h1 style='text-align: center; color: green;'>{well//10}</h1>", unsafe_allow_html=True)
      
  with kpi7:
      st.markdown("**Productivity Index**")
      st.markdown(f"<h1 style='text-align: center; color: green;'>{prod//10}</h1>", unsafe_allow_html=True)
  
  # Display berhan
  st.markdown("**Recent Emotional State**")
  st.line_chart(df[df['name'] == 'berhan'][['positive', 'negative']])  

st.title("Alert Monitoring")
st.markdown(f"<p1 style='text-align: center; color: yellow;'>{'Burak did not take a lunch break yet'}</p1>", unsafe_allow_html=True)
st.markdown(f"<p1 style='text-align: center; color: green;'>{'Total productivity index has risen by 14 points'}</p1>", unsafe_allow_html=True)
st.markdown(f"<p1 style='text-align: center; color: yellow;'>{'Taylan has large emotional fluctuations this week. It may be a good idea to visit workplace doctor.'}</p1>", unsafe_allow_html=True)
