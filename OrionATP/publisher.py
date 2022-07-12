import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as plt
import streamlit.components.v1 as components
from acquirer import final_df
from acquirer import selected
from acquirer import original_time
from trainer import detected_anomalies
from trainer import anomalydata

st.title("OrionATP")
fig = go.Figure()

# add the main values to the figure
fig.add_trace(go.Scatter(x = original_time['timestamp'], y = original_time['value'],   #blue line is original data over time
                             mode = 'lines',
                             marker =dict(color='blue'),
                             name = 'original_signal'))
fig.add_trace(go.Scatter(x = anomalydata['timestamp'], y = anomalydata['value'], mode = 'markers',
                             marker = dict(color='red'),
                             name = 'detected_anomaly'))
#fig.show() #graph figure with discrete anomalies highlighted in red
st.header("End-to-End Workflow for Unsupervised Anomaly Detection using Orion")
st.markdown("This is a visualization of the unsupervised anomaly detection performed using the Orion library. When automated using GitHub Actions, this updates on a regular basis using the new data it acquires. In this Streamlit App, we will walk through at a broad level how we aggregate and analyze data using the Orion library.")

st.header("Building the Pipeline")
st.markdown("Orion requires the data to be formatted in a very specific way, with the two columns of the dataframe being labelled 'timestamp' and 'value.' We set up the data to fit this format, then build a pipeline that doesn't vary much from the default ARIMA pipeline that's preprogrammed into the Orion library.")
st.dataframe(selected)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x = original_time['timestamp'], y = original_time['value'],   #blue line is original data over time
                             mode = 'lines',
                             marker =dict(color='blue'),
                             name = 'original_signal'))
st.plotly_chart(fig3)
st.markdown("This is what the data looks like when visualized. Next we will try to detect any potential anomalies.")
st.header("Detecting Anomalies")
st.markdown("Once the Orion pipeline is constructed, we fit the model to the data to train it and predict anomalous segments. The anomaly data is output into a dataframe representing the start and end of the anomalous sequence. Using a for loop, we split the sequence back into its original discrete data points that make it up in order to graph more effectively. Below are the points within the anomalous segments detected along with a visualization on the graph.")

st.dataframe(detected_anomalies)

st.dataframe(anomalydata)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x = anomalydata['timestamp'], y = anomalydata['value'], mode = 'markers',
                             marker = dict(color='red'),
                             name = 'detected_anomaly'))
st.plotly_chart(fig2)
st.markdown("As seen in the full data graphed earlier, the values tend to rise before falling a significant amount. Since the ARIMA model specifically uses a moving average, any sudden change will be detected as more anomalous. This must be the reason why the anomalous segments are detected during the greatest increases in the values of the data. If we find the dates corresponding with the timestamps, we can try to determine why anomalies were detected on those specific dates.")

st.header("Graphing the Data")
st.markdown("Finally, we graph the data. Depending on the type of data, plotly has different types of graphs of varying complexity, but this basic graph seems to work just fine for now. Discrete anomalous points are highlighted with red dots, while the timeseries data is graphed over time with the blue line.") 

st.plotly_chart(fig)

st.header("Conclusions")

st.markdown("It's obviously difficult to draw any sort of conclusions based on just this data set, but it seems like the Orion ARIMA pipeline is detecting anomalous segments during certain times of drastic increase in price. Obviously this script will change based on the Orion model used and the data set. There's also far too many variables that weren't taken into account for us to be able to draw any definitive conclusions about causation based off this data.")
