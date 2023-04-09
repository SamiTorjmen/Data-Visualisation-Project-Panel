importScripts("https://cdn.jsdelivr.net/pyodide/v0.22.1/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.4/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.4/dist/wheels/panel-0.14.4-py3-none-any.whl', 'pyodide-http==0.1.0', 'backcall==0.2.0', 'bleach==6.0.0', 'bokeh==2.4.3', 'certifi==2022.12.7', 'charset-normalizer==3.1.0', 'click==8.1.3', 'colorama==0.4.6', 'colorcet==3.0.1', 'cycler==0.11.0', 'debugpy==1.6.7', 'decorator==5.1.1', 'entrypoints==0.4', 'Flask==2.2.3', 'fonttools==4.38.0', 'gunicorn==20.1.0', 'holoviews==1.15.4', 'hvplot==0.8.3', 'idna==3.4', 'importlib-metadata==6.2.0', 'ipykernel==6.16.2', 'ipython==7.34.0', 'itsdangerous==2.1.2', 'jedi==0.18.2', 'Jinja2==3.1.2', 'joblib==1.2.0', 'jupyter_client==7.4.9', 'jupyter_core==4.12.0', 'kiwisolver==1.4.4', 'Markdown==3.4.3', 'MarkupSafe==2.1.2', 'matplotlib==3.5.3', 'matplotlib-inline==0.1.6', 'nest-asyncio==1.5.6', 'numpy==1.21.6', 'packaging==23.0', 'pandas==1.3.5', 'panel==0.14.4', 'param==1.13.0', 'parso==0.8.3', 'patsy==0.5.3', 'pickleshare==0.7.5', 'Pillow==9.5.0', 'plotly==5.14.1', 'prompt-toolkit==3.0.38', 'psutil==5.9.4', 'pyct==0.5.0', 'Pygments==2.14.0', 'pyparsing==3.0.9', 'python-dateutil==2.8.2', 'pytz==2023.3', 'pyviz-comms==2.2.1', 'PyYAML==6.0', 'pyzmq==25.0.2', 'requests==2.28.2', 'scikit-learn==1.0.2', 'scipy==1.7.3', 'seaborn==0.12.2', 'six==1.16.0', 'statsmodels==0.13.5', 'tenacity==8.2.2', 'threadpoolctl==3.1.0', 'tornado==6.2', 'tqdm==4.65.0', 'traitlets==5.9.0', 'typing_extensions==4.5.0', 'urllib3==1.26.15', 'wcwidth==0.2.6', 'webencodings==0.5.1', 'Werkzeug==2.2.3', 'zipp==3.15.0']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

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

df = pd.read_csv("data/StudentsPerformance.csv")
numeric_features = ['math score', 'reading score', 'writing score']
categoric_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
df['pass'] = df.apply(lambda row: 1 if row['math score'] >= 60 and row['reading score'] >= 60 and row['writing score'] >= 60 else 0, axis=1)


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import dashboard
from dashboard.plots import table_plotly
from dashboard.plots import pie_quali 
from dashboard.plots import histogram_quali 
from dashboard.plots import boxplot_quali_quanti 
from dashboard.plots import scatter_quanti_quanti
from dashboard.plots import plotting_target_feature
from dashboard.plots import corr_heatmap
from dashboard.plots import qqplot
from dashboard.plots import hist_residual
from dashboard.plots import qqplot_residual
from dashboard.plots import residual_fitted
from dashboard.plots import residual_leverage
from dashboard.plots import bivar_quanti_plot
from dashboard.plots import cross_heatmap
from dashboard.plots import ols_resid_plot
from dashboard.plots import confusion_matrix_heatmap
from dashboard.plots import plot_roc


from dashboard.tables import describe_quali_quanti
from dashboard.tables import filtered_dataframe
from dashboard.tables import evaluate_regression_model
from dashboard.tables import cross_tab
from dashboard.tables import chi2_tab
from dashboard.tables import report_to_df


from dashboard.model import model_history
from dashboard.model import model_cl_history

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

### Exploration Sidebar (Page 1)

# Cards
data_card = pn.Card(pn.Column(*checkboxes.values()), title='Data')
histogram_card = pn.Card(pn.Column(count), title='Histogram')
scatter_card = pn.Card(pn.Column(dashboard_fit_line_checkbox, abscisse_scatter, ordonnee_scatter), title='Scatter Plot')
box_card = pn.Card(pn.Column(quanti, quali), title='Box Plot')
target_card = pn.Card(pn.Column(quali_target), title='Target Plot')


# Sidebar 
exploration_sidebar = pn.Column('# Parameters\\n This section changes parameters for exploration plots',
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
modeling_sidebar = pn.Column('# Parameters\\n This section changes parameters for modeling plots',
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
analysis_sidebar = pn.Column('# Parameters\\n This section changes parameters for further analysis plots',
    quanti_quanti_card,
    quali_quali_card,
    quali_quanti_card,
    sizing_mode='stretch_width'
)

##### Define Main

### Main Exploration (Page 1)

# Cards
description = "This dataset contains information about the performance of students in various subjects. The data includes their scores in math, reading, and writing, as well as their gender, race/ethnicity, parental education, and whether they qualify for free/reduced lunch."
description_card = pn.Card(description, title='Description')

dataset_card = pn.Card(pn.Row(pn.Column('# Data ', description),
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

main_tabs = pn.Tabs(('Exploration', exploration_main_content),
                    ('Modeling', modeling_main_content),
                    ('Further Analysis', analysis_main_content))

def on_tab_change(event):

    if event.new == 0:

        exploration_sidebar.visible = True
        modeling_sidebar.visible = False
        analysis_sidebar.visible = False

    elif event.new == 1:


        exploration_sidebar.visible = False
        modeling_sidebar.visible = True
        analysis_sidebar.visible = False


    else:

        exploration_sidebar.visible = False
        modeling_sidebar.visible = False
        analysis_sidebar.visible = True


main_tabs.param.watch(on_tab_change, 'active')

##### Layout

template = pn.template.VanillaTemplate(
    
    # title
    title = "Student Performance in Exams",
    
    # sidebar
    sidebar = pn.Column(exploration_sidebar, modeling_sidebar, analysis_sidebar, sizing_mode='stretch_width'),
    
    # main
    main = main_tabs
)

#template.header.append(dark_mode_toggle)
##### Show Dashboard
template.servable()






await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.runPythonAsync(`
    import json

    state.curdoc.apply_json_patch(json.loads('${msg.patch}'), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads("""${msg.location}""")
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()