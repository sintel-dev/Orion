# OrionATP
**Acquirer, Transformer, and Publisher for Data Analysis and Visualization using the Orion library in Python**

_Created by Avi Shah_

Take time series data straight from API, detect anomalies using Orion library, and visualize using Streamlit. Scripts will run once a day as scheduled by the workflow for consistent analysis and progress tracking.

## Getting Started:

**Step 1:** Input API Key and name of time and value metric in api.txt, in that precise order. This is the file that the acquisition script will read from in order to collect and aggregate the data successfully. **Note:** You will need to know the labels of the columns for your time and value metric in order for this workflow to run successfully. This code currently supports json and csv file types. If the API request returns a different file type or an extremely nested json file, this may not work correctly.

**Step 2:** Input desired Pipeline and Hyperparameters into pipeline.json and hyperparameters.json respectively. This can simply be copy/pasted from Orion library's list of pipelines. These files exist so the pipeline and hyperparameters can be changed easily without having to edit the scripts themselves.

**Step 3:** When you fork this repo, deploy this app using https://streamlit.io/ and Python 3.7. There are fairly simple instructions on how to deploy an app on the Streamlit website. It's as easy as selecting a Github repo and choosing a Python version. We're currently trying to figure out if it is possible to omit this step, but as of now, the apps need to be deployed on Streamlit in order to be viewed on Github Pages.

**Step 4:** Go into your repository settings and enable GitHub Pages. This is so you don't have to use the Streamlit link when trying to view the visualization results. **The following is very important:** If you want your code to be available on Github Pages as well as Streamlit, you may have to make edits to the **index.html** file. This file will be specific to your app. The simplest thing to do is copy and paste the HTML source code directly from Streamlit by inspecting it through your browser. Only a few lines need to be changed however. You should replace our GitHub paths with your own, as well as the title "OrionATP" if you want. Use Control + F to find these specific lines more easily.

**Step 5:** Manually trigger the workflow or wait for it to trigger on schedule. You should be all set and able to view the results of your anomaly detection and visualization! The publisher has some text already that is specific to our demonstration (namely it mentions using specifically the ARIMA model and a few other minor things). You may want to edit this to suit your needs.
