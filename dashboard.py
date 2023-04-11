import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns

sns.set_style('whitegrid')

import panel as pn
from panel.interact import interact
pn.extension('plotly') # Interactive tables

import hvplot.pandas # Interactive dataframes

import holoviews as hv
from bokeh.events import Event
hv.extension('bokeh')

df = pd.read_csv("https://raw.githubusercontent.com/SamiTorjmen/Data-Visualisation-Project-Panel/master/data/StudentsPerformance.csv")
numeric_features = ['math score', 'reading score', 'writing score']
categoric_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
df['pass'] = df.apply(lambda row: 1 if row['math score'] >= 60 and row['reading score'] >= 60 and row['writing score'] >= 60 else 0, axis=1)


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import dashboard_dataviz_panel as dashboard
from dashboard_dataviz_panel.plots import table_plotly
from dashboard_dataviz_panel.plots import pie_quali 
from dashboard_dataviz_panel.plots import histogram_quali 
from dashboard_dataviz_panel.plots import boxplot_quali_quanti 
from dashboard_dataviz_panel.plots import scatter_quanti_quanti
from dashboard_dataviz_panel.plots import plotting_target_feature
from dashboard_dataviz_panel.plots import corr_heatmap
from dashboard_dataviz_panel.plots import qqplot
from dashboard_dataviz_panel.plots import hist_residual
from dashboard_dataviz_panel.plots import qqplot_residual
from dashboard_dataviz_panel.plots import residual_fitted
from dashboard_dataviz_panel.plots import residual_leverage
from dashboard_dataviz_panel.plots import bivar_quanti_plot
from dashboard_dataviz_panel.plots import cross_heatmap
from dashboard_dataviz_panel.plots import ols_resid_plot
from dashboard_dataviz_panel.plots import confusion_matrix_heatmap
from dashboard_dataviz_panel.plots import plot_roc


from dashboard_dataviz_panel.tables import describe_quali_quanti
from dashboard_dataviz_panel.tables import filtered_dataframe
from dashboard_dataviz_panel.tables import evaluate_regression_model
from dashboard_dataviz_panel.tables import cross_tab
from dashboard_dataviz_panel.tables import chi2_tab
from dashboard_dataviz_panel.tables import report_to_df


from dashboard_dataviz_panel.model import model_history
from dashboard_dataviz_panel.model import model_cl_history

pn.config.sizing_mode = "stretch_width"


reg_list = [
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
]

cl_list= [
    LogisticRegression,
    RandomForestClassifier,
    KNeighborsClassifier,
    SVC
]

##### Create widgets

### Exploration widgets (Page 1)

# Dataset
checked_columns = ['lunch', 'race/ethnicity','test_preparation_course','math score','reading score','writing score','target_name']
checkboxes = {col: pn.widgets.Checkbox(name=col, value=True) if col in checked_columns else  pn.widgets.Checkbox(name=col, value=False) for col in df.columns}

# Histogram
count = pn.widgets.Select(name='feature',options=[col for col in df.columns], value='parental level of education')

# Scatter plot 
abscisse_scatter = pn.widgets.Select(name='x', options=numeric_features, value='reading score')
ordonnee_scatter = pn.widgets.Select(name='y', options=numeric_features, value='writing score')
dashboard_fit_line_checkbox = pn.widgets.Checkbox(name='fit line')

# Box plot
quanti = pn.widgets.Select(name='numeric feature', options=numeric_features)
quali = pn.widgets.Select(name='categorical feature', options=categoric_features, value='parental level of education')

# Target Plot
quali_target = pn.widgets.Select(name='categorical feature', options=categoric_features, value='parental level of education')

### Modeling Widget (Page 2)

# Regression
target_widget =  pn.widgets.Select(name='target', options=numeric_features, value='writing score')
model_name_widget = pn.widgets.Select(name='model', options=reg_list, value=LinearRegression)


# Classification
model_name_cl_widget = pn.widgets.Select(name='classification model', options=cl_list, value=LogisticRegression)
color_confusion = pn.widgets.Select(name='Matrix color', options=px.colors.named_colorscales(), value='bupu')

### Analysis Widget (Page 3)

# Quanti/Quanti
color1 = pn.widgets.Select(name='color', options=px.colors.named_colorscales(), value='magma')
quanti1_corr = pn.widgets.Select(name='x',options=numeric_features, value = 'reading score')
quanti2_corr = pn.widgets.Select(name='y',options=numeric_features, value = 'writing score')

# Quali/Quali
color2 = pn.widgets.Select(name='color', options=px.colors.named_colorscales(), value='redor')
quali1_cross = pn.widgets.Select(name='quali 1',options=categoric_features, value = 'parental level of education')
quali2_cross = pn.widgets.Select(name='quali 2',options=categoric_features, value = 'lunch')


# Q-Q Plot
quanti_qq = pn.widgets.Select(name='numeric feature', options=numeric_features)
quali_qq = pn.widgets.Select(name='categorical feature', options=categoric_features, value='parental level of education')
modality_qq = pn.widgets.Select(name='modality', options=df[quali_qq.params.args[0].value].unique().tolist())


def update_modality_options(event):
    selected_quali = quali_qq.value
    selected_modality = modality_qq.value
    modality_qq.options = df[selected_quali].unique().tolist()
    if selected_modality not in modality_qq.options:
        modality_qq.value = modality_qq.options[0]
    else:
        modality_qq.value = selected_modality

quali_qq.param.watch(update_modality_options, 'value')


##### Define reactive elements

### Reactive elements for Exploration (Page 1)

dataset = pn.bind(filtered_dataframe, df=df, **checkboxes)
histogram = pn.bind(histogram_quali,quali=count,df=df)
scatter_plot = pn.bind(scatter_quanti_quanti, x=abscisse_scatter, y=ordonnee_scatter, df=df, checkbox=dashboard_fit_line_checkbox)
box_plot = pn.bind(boxplot_quali_quanti, quanti=quanti, quali=quali, df=df)
describe_table = pn.bind(describe_quali_quanti, quali=quali, quanti=quanti, df=df)
target_plot = pn.bind(plotting_target_feature, quali=quali_target,df=df)


### Reactive elements for Modeling (Page 2)

# Regression

def update_reg_history(target, model):
    return model_history(df=df, target=target, model=model)

reg_history = pn.bind(update_reg_history, target=target_widget, model=model_name_widget)

evaluate_reg_table = pn.bind(evaluate_regression_model,history=reg_history)
residual_fitted_plot = pn.bind(residual_fitted, history=reg_history)
qqplot_residual_plot = pn.bind(qqplot_residual, history=reg_history)
scale_location_plot = pn.bind(residual_fitted, history=reg_history, root=True)
residual_leverage_plot = pn.bind(residual_leverage, history=reg_history)

# Classification
def update_cl_history(model_cl):
    return model_cl_history(df=df, model_cl=model_cl)

cl_classification = pn.bind(update_cl_history, model_cl=model_name_cl_widget)
evaluate_cl_table = pn.bind(report_to_df,classification=cl_classification)

confusion_plot = pn.bind(confusion_matrix_heatmap, classification=cl_classification,color=color_confusion)

roc = pn.bind(plot_roc, classification=cl_classification)

### Reactive elements for Analysis (Page 3)
corr_plot = pn.bind(corr_heatmap, df=df, quanti1=quanti1_corr,quanti2=quanti2_corr, color=color1)
joint_plot = pn.bind(bivar_quanti_plot, df=df, quanti1=quanti1_corr, quanti2=quanti2_corr)

cross_table = pn.bind(cross_tab, df=df, quali1=quali1_cross, quali2=quali2_cross)
chi2_table = pn.bind(chi2_tab, df=df, quali1=quali1_cross, quali2=quali2_cross)
cross_heatmap_plot = pn.bind(cross_heatmap, df=df, quali1=quali1_cross, quali2=quali2_cross, color=color2)

box_plot2 = pn.bind(boxplot_quali_quanti, quanti=quanti_qq, quali=quali_qq, df=df)
qq_plot = pn.bind(qqplot, quali=quali_qq, quanti=quanti_qq, modality=modality_qq, df=df)
ols_plot = pn.bind(ols_resid_plot, df=df, quanti=quanti_qq, quali=quali_qq)

##### Define Sidebar

#### Home Sidebar (Page 0)

sidebar_desciption = pn.pane.Markdown("""
###  Exploration Sidebar (Page 1)
 Checkboxes to select data

 Histogram plot

 Scatter plot

 Box plot

 Target plot

###  Modeling Sidebar (Page 2)
 Regression plot

 Classification plot

###  Analysis Sidebar (Page 3)
 Quantitative vs Quantitative plot

 Qualitative vs Qualitative plot

 Qualitative vs Quantitative plot
""")
home_sidebar =pn.Column(sidebar_desciption)


### Exploration Sidebar (Page 1)

# Cards
data_card = pn.Card(pn.Column(*checkboxes.values()), title='Data')
histogram_card = pn.Card(pn.Column(count), title='Histogram')
scatter_card = pn.Card(pn.Column(dashboard_fit_line_checkbox, abscisse_scatter, ordonnee_scatter), title='Scatter Plot')
box_card = pn.Card(pn.Column(quanti, quali), title='Box Plot')
target_card = pn.Card(pn.Column(quali_target), title='Target Plot')


# Sidebar 
exploration_sidebar = pn.Column('# Parameters\n This section changes parameters for exploration plots',
    data_card,
    histogram_card,
    scatter_card,
    box_card,
    target_card,
    sizing_mode='stretch_width',
)

### Modeling Sidebar (Page 2)

# Cards
regression_card = pn.Card(pn.Column(model_name_widget,target_widget), title='Regression',sizing_mode = "stretch_width")

classification_card = pn.Card(pn.Column(model_name_cl_widget, color_confusion), title='Classification',sizing_mode = "stretch_width")



# Sidebar 
modeling_sidebar = pn.Column('# Parameters\n This section changes parameters for modeling plots',
    regression_card,
    classification_card,
    sizing_mode='stretch_width'
)


### Analysis Sidebar (Page 3)

# Cards
quanti_quanti_card = pn.Card(pn.Column(color1,quanti1_corr,quanti2_corr), title='Quantitative vs Quantitative')
quali_quali_card = pn.Card(pn.Column(color2,quali1_cross, quali2_cross), title='Qualitative vs Qualitative')
quali_quanti_card = pn.Card(pn.Column(quanti_qq,pn.Column(quali_qq, modality_qq)), title='Qualitative vs Quantitative')

# Sidebar 
analysis_sidebar = pn.Column('# Parameters\n This section changes parameters for further analysis plots',
    quanti_quanti_card,
    quali_quali_card,
    quali_quanti_card,
    sizing_mode='stretch_width'
)

### Home Page (Page 0)
home_description = pn.pane.Markdown("""
This dashboard allows you to explore the relationships between various factors that affect students' academic performance, including their gender, race/ethnicity, parental education, and whether they qualify for free/reduced lunch.

You can navigate between the different tabs to view visualizations and analysis of the data.

### Exploration

This section allows users to explore the dataset using histograms, scatter plots, box plots, and target plots. Users can customize the visualizations by selecting features and adjusting other settings through the sidebar widgets.

### Modeling

This section incorporates machine learning models for regression and classification tasks. Users can select different models and target variables, and the dashboard displays evaluation metrics, residual plots, Q-Q plots, scale-location plots, leverage plots, confusion matrices, and ROC curves.

### Analysis

This section provides additional statistical analysis, including correlation matrices, bivariate plots, contingency tables, chi-square tests, contingency heatmaps, box plots, and OLS residuals. Users can select features and compare their relationships using the sidebar widgets.
""")

# Content
home_main_content = pn.Column(pn.Row("# Welcome to the Student Performance Dashboard"),
                              pn.Row(home_description))

### Main Exploration (Page 1)

# Cards
# description = "This dataset contains information about the performance of students in various subjects. The data includes their scores in math, reading, and writing, as well as their gender, race/ethnicity, parental education, and whether they qualify for free/reduced lunch."
# description_card = pn.Card(description, title='Description')

dataset_card = pn.Card(pn.Row(#pn.Column('# Data ', description),
                            pn.Column(dataset)),
                    title='Description')

boxplot_card = pn.Row(pn.Card(describe_table, title='Describe Table'),
                    pn.Card(box_plot, title='Box Plot'))


scatter_hist_card = pn.Row(pn.Card(histogram, title='Histogram'), 
                        pn.Card(scatter_plot, title='Scatter Plot'))
target_card = pn.Card(target_plot, title='Target Plot')

# Content
exploration_main_content = pn.Column(
        pn.Row(dataset_card),
        pn.Row(scatter_hist_card),
        pn.Row(boxplot_card),
        pn.Row(target_card),
        sizing_mode='stretch_width')


### Main Modeling (Page 2)

# Cards
evaluate_table_card = pn.Card(evaluate_reg_table, title="Evaluation")
residual_fitted_card = pn.Card(residual_fitted_plot ,title="Residual Plot")
qqplot_residual_card = pn.Card(qqplot_residual_plot,title="Normal Q-Q")
scale_location_card = pn.Card(scale_location_plot, title="Scale Location")
residual_leverage_card = pn.Card(residual_leverage_plot, title="Residuals vs Leverage")

# Regroup cards
regression_card = pn.Card(pn.Row(evaluate_table_card),
                        pn.Row(residual_fitted_card,qqplot_residual_card),
                        pn.Row(scale_location_card,residual_leverage_card),      
                        title = 'Regression')

## Classification

evaluate_cl_card = pn.Card(evaluate_cl_table, title="Evaluation Table")
confusion_card = pn.Card(confusion_plot, title="Confusion Matrix")
roc_card = pn.Card(roc, title='ROC')

classification_card = pn.Card(pn.Row(evaluate_cl_card),
                            pn.Row(confusion_card,roc_card),
                            title='Classification')


# Content
modeling_main_content = pn.Column(pn.Row(regression_card),
                                pn.Row(classification_card),                                  
                                sizing_mode='stretch_width')


### Main Analysis(Page 3)

# Cards
corr_card = pn.Card(corr_plot, title='Person Correlation Matrix')
joint_card = pn.Card(joint_plot, title='Bivariate Plot')

cross_card = pn.Card(cross_table, title='Contingency Table')
chi2_card = pn.Card(chi2_table, title='Chi2 Test')
cross_heatmap_card = pn.Card(cross_heatmap_plot, title='Contingency Heatmap')

boxplot_card = pn.Card(box_plot2, title='Box Plot')
qq_card = pn.Card(qq_plot, title='Q-Q Plot')
ols_card = pn.Card(ols_plot, title='OLS Residuals')


quanti_quanti_card = pn.Card(pn.Row(corr_card,joint_card),
                            title=f'Statistic Dependency {quanti1_corr.params.args[0].value} vs {quanti2_corr.params.args[0].value} (quantitative/quantitative)')

quali_quali_card = pn.Card(pn.Row(pn.Column(cross_card,chi2_card),
                                cross_heatmap_card),
                        title=f'Statistic Dependency {quali1_cross.params.args[0].value} vs {quali2_cross.params.args[0].value} (qualitative/qualitative)')


quali_quanti_card = pn.Card(pn.Row(boxplot_card),
                            pn.Row(ols_card,qq_card),
                            title=f'Statistic Dependency {quali_qq.params.args[0].value} vs {quanti_qq.params.args[0].value} (qualitative/quantitative)')


# Content
analysis_main_content = pn.Column(pn.Row(quanti_quanti_card),
                                pn.Row(quali_quali_card),
                                pn.Row(quali_quanti_card), 
                                sizing_mode='stretch_width')


##### Create Callback to change sidebar content

main_tabs = pn.Tabs(('Home',home_main_content),
                    ('Exploration', exploration_main_content),
                    ('Modeling', modeling_main_content),
                    ('Further Analysis', analysis_main_content))

def on_tab_change(event): 

    if event.new == 0:
        home_sidebar.visible = True
        exploration_sidebar.visible = False
        modeling_sidebar.visible = False
        analysis_sidebar.visible = False

    elif event.new == 1:

        home_sidebar.visible = False
        exploration_sidebar.visible = True
        modeling_sidebar.visible = False
        analysis_sidebar.visible = False


    elif event.new == 2:

        home_sidebar.visible=False
        exploration_sidebar.visible = False
        modeling_sidebar.visible = True
        analysis_sidebar.visible = False
    
    else:

        home_sidebar.visible=False
        exploration_sidebar.visible = False
        modeling_sidebar.visible = False
        analysis_sidebar.visible = True

main_tabs.param.watch(on_tab_change, 'active')

exploration_sidebar.visible = False
modeling_sidebar.visible = False
analysis_sidebar.visible = False

##### Layout

template = pn.template.FastListTemplate(
    
    # title
    title = "Student Performance in Exams",
    
    # sidebar
    sidebar = pn.Column(home_sidebar,exploration_sidebar, modeling_sidebar, analysis_sidebar, sizing_mode='stretch_width'),
    
    # main
    main = main_tabs
)

#template.header.append(dark_mode_toggle)
##### Show Dashboard
template.servable()




