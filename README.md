# 📚 Project description

The project purpose is to answer a business question related to [the following dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) (Student's performance in exams) : 

<strong>How can we explain and predict both math, reading and writing scores of high school students in the United States ?</strong>


# 🌐 GitHub Pages

Visit our [dashboard on GitHub Pages](https://samitorjmen.github.io/Data-Visualisation-Project-Panel/) and explore the interactive data visualization. 🕐 Please be a little bit patient as it may take some time to set up the Pyodide environment.


# 📊 Interactive Data Visualization

The project's primary objective is to create an interactive data visualization dashboard to analyze student performance based on various demographic and socioeconomic factors. This dashboard aims to provide insights into the relationships between student performance and factors such as gender, race/ethnicity, parental level of education, lunch, and test preparation course. By exploring these relationships, users can gain a better understanding of how these factors impact academic achievement and use this information to make informed decisions in educational policy and practice.

## 📝 Code Overview

The code provided imports necessary libraries, loads the data, and preprocesses it to create an interactive dashboard. The dashboard consists of several visualization techniques and machine learning models to analyze the data and generate insights. The code can be divided into the following sections:

### 📥 Data Import and Preprocessing

The data is imported from a CSV file and stored in a Pandas dataframe. It contains information on students' math, reading, and writing scores, gender, race/ethnicity, parental level of education, lunch, and test preparation course. A new binary feature, 'pass', is created to indicate whether a student has passed all three subjects with a score of 60 or above.

### 📈 Visualization Libraries and Panel Extensions

The code uses Plotly, Matplotlib, Seaborn, HoloViews, and Panel for data visualization and interactive elements. These libraries are imported, and necessary extensions are enabled for smooth rendering and interactivity.

### 🎛️ Dashboard Components and Widgets

The dashboard is designed using Panel, a high-level app and dashboarding solution for Python. Various widgets like checkboxes, dropdowns, and sliders are created to allow users to interact with the dashboard and customize the visualizations.

### ⚡ Reactive Elements

Reactive elements are defined using Panel's bind function, which enables the visualizations to update automatically when the user interacts with the widgets. These reactive elements are created for each visualization and machine learning model used in the dashboard.

### 📐 Dashboard Layout

The dashboard is organized into three main sections: Exploration, Modeling, and Analysis. Each section has a sidebar containing widgets for user interaction and a main content area with visualizations and results.

### 🔍 Exploration

This section allows users to explore the dataset using histograms, scatter plots, box plots, and target plots. Users can customize the visualizations by selecting features and adjusting other settings through the sidebar widgets.

### 🤖 Modeling

This section incorporates machine learning models for regression and classification tasks. Users can select different models and target variables, and the dashboard displays evaluation metrics, residual plots, Q-Q plots, scale-location plots, leverage plots, confusion matrices, and ROC curves.

### 📊 Analysis

This section provides additional statistical analysis, including correlation matrices, bivariate plots, contingency tables, chi-square tests, contingency heatmaps, box plots, and OLS residuals. Users can select features and compare their relationships using the sidebar widgets.

## 🚀 How to Serve the App

To serve the app locally, follow these steps:

1. 📥 Clone or download the repository to your local machine.

2. 📍 Navigate to the project directory in your terminal or command prompt.
<pre>
```
cd Data-Visualisation-Projet-Panel
```
</pre>

3. 📦 Install the necessary libraries by running the following command in your terminal or command prompt:

<pre>
```bash
pip install -r requirements.txt
```
</pre>

4. 🎮 Run the following command to serve the app:
<pre>
```bash
panel serve dashboard.py
```
</pre>

5. 🌐 Open your web browser and navigate to the URL provided in the terminal or command prompt, usually http://localhost:5006/app. You should now see the interactive data visualization dashboard.

6. 🎛️ Use the sidebar widgets to interact with the dashboard and explore the visualizations.

# 🧩 Module dashboard-dataviz-panel

This package provides a set of functions and classes to help create interactive data visualization dashboards.

## ⚙️ Installation

To install the package, simply run:

<pre>
```bash
pip install dashboard-dataviz-panel
```
</pre>


## 📚 Documentation

For detailed documentation, please visit our [official documentation](https://pypi.org/project/dashboard-dataviz-panel/0.1.3/).

## 📦 Module

Our module contains various functions and classes to assist in the creation of dashboards. You can learn more about the module [here](https://github.com/SamiTorjmen/Data-Visualisation-Project-Panel).
