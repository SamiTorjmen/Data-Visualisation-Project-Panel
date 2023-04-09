############# HELPERS
import plotly.express as px


import re
import plotly.express as px


def rgb_to_hex(rgb):
    """
    Converts RGB values to hexadecimal format.

    Args: rgb_tuple
    
        r (int): The red component (0-255).
        g (int): The green component (0-255).
        b (int): The blue component (0-255).

    Returns:
        str: The hexadecimal representation of the RGB color.

    """
    
    r, g, b = rgb
    return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)


def extract_rgb(rgb_string):
    """
    Extracts RGB values from a string in the format "rgb(r, g, b)"
    Returns a tuple of integers (r, g, b)
    """
    # Extract the numbers 
    rgb_numbers = re.findall(r'\d+', rgb_string)
    
    # Return a tuple
    return tuple(map(int, rgb_numbers))


def plotly_to_plt_colors(rgb_string):
    return rgb_to_hex(extract_rgb(rgb_string))

def color_s(column,apply=True):
    ''' 
    This function apply colors for modalities of a column
    '''
    
    if apply:
        
        if (column=='gender'): return([px.colors.qualitative.Safe[1], px.colors.qualitative.Safe[0]])
    if (column=='pass'): return([px.colors.qualitative.Safe[3], px.colors.qualitative.Safe[5]])


    return px.colors.qualitative.Safe


def categarray(column):
    '''
    This function order modalities of a column
    '''
    if (column=='gender'): return(['male','female'])
    if (column=='race/ethnicity'): return (['group A', 'group B', 'group C', 'group D', 'group E'])
    if (column=='parental level of education'): return(['some high school', 'high school','some college',"associate's degree","bachelor's degree", "master's degree"])
    if (column=='pass'): return(['0','1'])

############ Tables 
import pandas as pd
import panel as pn
from scipy import stats
from sklearn.metrics import classification_report

pn.extension('plotly')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Table:
    def __init__(self, df):
        self.df = df

    def to_panel(self):
        df = self.df
                
        if df.index.name:
            width = str(100/(len(df.columns)+1))+'%'
            widths = {col: width for col in df.columns}
            widths[df.index.name] = width

        else:
            width = str(100/(len(df.columns)))+'%'
            widths = {col: width for col in df.columns}

        # else:
        #     widths['index'] = width

        tabulator = pn.widgets.Tabulator(df,
                                         page_size=10,
                                         text_align='left',
                                         header_align='center',
                                         hidden_columns=['index'],
                                         widths=widths)
        return tabulator

#pn.config.sizing_mode = 'stretch_width'


def describe_quali_quanti(quali, quanti, df):
    """
    display mean, count, std of quantitative for each category of the variable qualitative
    --------------------------------------------------------------------------------------
    quali -> string. example 'gender'
    quanti -> string. example "math score"
    df -> DataFrame
    """
    
    df_g= df.groupby([quali])[quanti].agg(['count', 'mean', 'std']).sort_values(by='mean', ascending=False)
    # print('average / standard ', quali)
    # print(df)
    # print('')
    
    return Table(df_g).to_panel()



def cross_tab(df, quali1, quali2):

    crosstab = pd.crosstab(df[quali1], df[quali2])

    return Table(crosstab).to_panel()

def chi2_tab(df, quali1, quali2):

    crosstab = pd.crosstab(df[quali1], df[quali2])
    chi2_test = stats.chi2_contingency(crosstab)

    # Extract the results
    chi2, p_value, dof, expected = chi2_test

    # Create a dictionary to store the results
    results = {
        "Chi-Square": [chi2],
        "p-value": [p_value],
        "Degrees of Freedom": [dof]
    }

    # Create a DataFrame from the dictionary
    chi2_df = pd.DataFrame(results)

    return Table(chi2_df).to_panel()


def filtered_dataframe(df, **checkboxes_values):
    '''
    A reactive function to filter the dataframe based on the checked checkboxes
    ---------------------------------------------------------------------------
    df -> DataFrame
    checkboxes_values -> Dict 
    '''
    selected_columns = [col for col, value in checkboxes_values.items() if value]
    return Table(df[selected_columns]).to_panel()


def evaluate_regression_model(history):

    # Calculate metrics
    mse = mean_squared_error(history.y_test, history.y_pred)
    rmse = mean_squared_error(history.y_test, history.y_pred, squared=False)
    mae = mean_absolute_error(history.y_test, history.y_pred)
    r2 = r2_score(history.y_test, history.y_pred)
    # Create a dictionary with the results
    results = {'R2 Score': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    # Create a DataFrame from the dictionary and return it
    column = str(history.model)
    eval_df = pd.DataFrame.from_dict(results, orient='index', columns=[column])
    eval_df.insert(0,'metric',eval_df.index)
    return Table(eval_df).to_panel()


# Classification

def report_to_df(classification):
    
    report = classification_report(classification.y_test_cl, classification.y_pred_cl, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.rename_axis('Class', inplace=True)
    
    return Table(df.head(len(classification.classes))).to_panel()

#################### Plots

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc


import matplotlib.pyplot as plt
import seaborn as sns

import panel as pn
pn.extension('plotly')

#pn.config.sizing_mode = "stretch_width"

##### Table

def table_plotly(df):
    # create a table using Plotly
    table = go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='white',
            align='center',
            line_color='darkslategray',
            font=dict(color='darkslategray', size=12)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='white',
            align='center',
            line_color='darkslategray',
            font=dict(color='darkslategray', size=11)
        )
    )
    df_plotly = go.Figure(data=table)
    df_plotly.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return df_plotly


##### Univariée

## Pie
def pie_quali(quali,df):
    """
    plot a pie of categorical  variable
    --------------------------------------------------------
    quali -> array of string. example 'gender'
    df -> DataFrame
    """

    # Group the DataFrame by the categorical variable and count the number of unique occurrences
    count = df.groupby(quali).nunique()

    # create the pie using Plotly
    fig = px.pie(count, 
                names=count.index,
                values='id',
                title=(f"Distribution of {quali}"),
                color_discrete_sequence=color_s(quali))

    # rearange axes
    fig.update_xaxes(categoryorder='array', categoryarray=categarray(quali))

    # show the histogram
    return fig

## Histogram
def histogram_quali(quali,df):
    """
    plot a histogram of categorical  variable
    --------------------------------------------------------
    quali -> array of string. example 'gender'
    df -> DataFrame
    """
    # create the histogram using Plotly
    fig = px.histogram(df, 
                    x=quali,
                    title=(f"Distribution of {quali}"),
                    color=quali,
                    color_discrete_sequence=color_s(quali))

    # Set the font sizes for the axis labels
    fig.update_layout(xaxis=dict(title=dict(font=dict(size=20)),
                                 showline=True,
                                 linewidth=1,
                                 linecolor='gray',
                                 mirror=True),

                      yaxis=dict(title=dict(font=dict(size=20)),
                                 gridcolor='whitesmoke',
                                 showline=True,
                                 linewidth=1,
                                 linecolor='gray',
                                 mirror=True),
                      plot_bgcolor='white')

    # rearange axes
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=categarray(quali))
    
    
    # show the histogram
   
    return fig

##### Bivariée

## Box Plot
def boxplot_quali_quanti(quali, quanti, df):
    """
    plot a boxplot between categorical et numerical variable
    --------------------------------------------------------
    quali -> array of string. example ['diplome', 'sexe']
    quanti -> string. example "salaire"
    df -> DataFrame
    """

    # Create the figure with Plotly Express
    fig = px.box(df, 
                 x=quali, 
                 y=quanti, 
                 color=quali, 
                 color_discrete_sequence=color_s(quali),
                 title=f"{quanti} vs {quali}")

    # Set the font sizes for the axis labels
    fig.update_layout(xaxis=dict(title=dict(font=dict(size=20)),
                                 showline=True,
                                 linewidth=1,
                                 linecolor='gray',
                                 mirror=True),

                      yaxis=dict(title=dict(font=dict(size=20)),
                                 gridcolor='whitesmoke',
                                 showline=True,
                                 linewidth=1,
                                 linecolor='gray',
                                 mirror=True),
                      plot_bgcolor='white')

    fig.update_xaxes(categoryorder='array', categoryarray=categarray(quali))
    
    fig
    return fig
           
## Scatter Plot
def scatter_quanti_quanti(x, y, df, checkbox):
    scatter = df.hvplot.scatter(x, y).opts(width=450)
    
    if checkbox:
        scatter.opts(line_color='black')
        return scatter * hv.Slope.from_scatter(scatter).opts(line_color='pink')
    else:
        scatter.opts(line_color='black')
        return scatter

## Target Plot
def plotting_target_feature(quali, df):
    df['target_name'] = df['pass'].map({0: 'Fail', 1: 'Pass'})
    # Figure initiation
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18,12))

    ### Number of occurrences per categoty - target pair 
    order = categarray(quali) # Get the order of the categorical values
    
    # Set the color palette
    colors = list(map(lambda x: plotly_to_plt_colors(x), color_s(quali,apply=False)))
    sns.set_palette(sns.color_palette(colors)) 
    
    ax1 = sns.countplot(x=quali, hue="target_name", data=df, order=order, ax=axes[0])
    # X-axis Label
    ax1.set_xlabel(quali, fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    # Y-axis Label
    ax1.set_ylabel('Number of occurrences', fontsize=14)
    # Adding Super Title (One for a whole figure)
    fig.suptitle('Graphiques '+quali + ' par rapport à la réussite' , fontsize=18)
    # Setting Legend location 
    ax1.legend(loc=1)

    ### Adding percents over bars
    # Getting heights of our bars
    height = [p.get_height() for p in ax1.patches]
    # Counting number of bar groups 
    ncol = int(len(height)/2)
    # Counting total height of groups
    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
    # Looping through bars
    for i, p in enumerate(ax1.patches):    
        # Adding percentages
        ax1.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 10,
                '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14) 


    ### Survived percentage for every value of feature
    ax2 = sns.pointplot(x=quali, y='pass', data=df, order=order, ax=axes[1])
    # X-axis Label
    ax2.set_xlabel(quali, fontsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    # Y-axis Label
    ax2.set_ylabel('Pourcentage de réussite', fontsize=14)
    
    plt.close()
    
    return pn.pane.Matplotlib(fig, sizing_mode='stretch_both')

## Heatmap
def corr_heatmap(df, quanti1, quanti2, color):

    # Calculate the correlation matrix
    corrmat = df[[quanti1,quanti2]].corr(method='pearson')

    # Create a Plotly heatmap
    fig = ff.create_annotated_heatmap(
        z=corrmat.values,
        x=list(corrmat.columns),
        y=list(corrmat.index),
        annotation_text=corrmat.round(2).values,
        colorscale=color,
        zmin=0,
        zmax=1,
        showscale=True
    )

    # Update layout
    fig.update_layout(
        title='Pearson Correlation of Features',
        xaxis=dict(side='bottom', tickangle=0),
        yaxis=dict(autorange='reversed')
    )

    return fig



def bivar_quanti_plot(df, quanti1, quanti2):
    fig = sns.jointplot(x=quanti1, y=quanti2, data=df, color='red', kind='kde').fig
    plt.close()
    return pn.pane.Matplotlib(fig, sizing_mode='stretch_both')

def cross_heatmap(df,quali1, quali2, color):

    crosstab = pd.crosstab(df[quali1], df[quali2], normalize='index')

    # Create a Plotly heatmap
    fig = ff.create_annotated_heatmap(
        z=crosstab.values,
        x=list(crosstab.columns),
        y=list(crosstab.index),
        annotation_text=crosstab.round(2).values,
        colorscale=color,
        zmin=0,
        zmax=1,
        showscale=True
    )

    # Update layout
    fig.update_layout(
        title='Pearson Correlation of Features',
        xaxis=dict(side='bottom', tickangle=0),
        yaxis=dict(autorange='reversed')
    )

    return fig

def ols_resid_plot(df, quali, quanti):
    le = LabelEncoder()
    data = df.copy()
    data[quali] = le.fit_transform(df[quali])
    
    results = ols(f"Q('{quanti}') ~ Q('{quali}')", data=data).fit()
    residuals = results.resid

    residual_df = pd.DataFrame({f'{quali}': data[quali], 'Residuals OLS': residuals})
    scatter = residual_df.hvplot.scatter(f'{quali}', 'Residuals OLS')

    scatter.opts(line_color='black')

    return scatter * hv.HLine(0).opts(color='red', line_width=1)


## Q-Q Plot
def qqplot(quali, quanti, modality, df):   
    
   

    selected_data = df[quanti][df[quali] == modality]
    qq_points = stats.probplot(selected_data, fit=False)
    qq_df = pd.DataFrame({'x': qq_points[0], 'y': qq_points[1]})

    scatter = qq_df.hvplot.scatter('x', 'y',  title="Q-Q Plot for '{}' with '{}' = '{}'".format(quanti, quali, modality))

    scatter.opts(line_color='black')

    return scatter * hv.Slope.from_scatter(scatter).opts(line_color='red',line_width=1)
    

## Residuals

def hist_residual(history):

    # create the histogram using Plotly
    fig = px.histogram( 
                    x=history.residuals,
                    title=(f"Distribution of residuals for {str(history.model)}"),
                    #color=residuals,
                    color_discrete_sequence=px.colors.qualitative.Safe[2:])

    # Set the font sizes for the axis labels
    fig.update_layout(xaxis=dict(title=dict(text='Residuals',font=dict(size=20)),
                                 showline=True,
                                 linewidth=1,
                                 linecolor='gray',
                                 mirror=True),

                      yaxis=dict(title=dict(font=dict(size=20)),
                                 gridcolor='whitesmoke',
                                 showline=True,
                                 linewidth=1,
                                 linecolor='gray',
                                 mirror=True),
                      plot_bgcolor='white')

    return fig

def residual_fitted(history,root=False):

    if not root:
        residual_df = pd.DataFrame({'Predicted Values': history.y_pred, 'Residuals': history.residuals})
        scatter = residual_df.hvplot.scatter('Predicted Values', 'Residuals')

    else:
        residual_df =  pd.DataFrame({'Predicted values': history.y_pred, 'Root Standardized Residuals': history.residuals.apply(lambda x: np.sqrt(np.abs(x)))})
        scatter = residual_df.hvplot.scatter('Predicted values', 'Root Standardized Residuals')

    scatter.opts(line_color='black')
    
    return scatter * hv.Slope.from_scatter(scatter).opts(line_color='red',line_width=1)

def qqplot_residual(history):

    
    qq_points = stats.probplot(history.residuals, fit=False)
    qq_df = pd.DataFrame({'Theorical Quantiles': qq_points[0], 'Standardized residuals': qq_points[1]})
    
    scatter = qq_df.hvplot.scatter('Theorical Quantiles', 'Standardized residuals')
    
    scatter.opts(line_color='black')
    
    return scatter * hv.Slope.from_scatter(scatter).opts(line_color='red',line_width=1)

def residual_leverage(history):
    model = sm.regression.linear_model.OLS(history.y_train, sm.add_constant(history.X_train)).fit()
    influence = model.get_influence()

    leverage = influence.hat_matrix_diag
    cooks_distance = influence.cooks_distance[0]
    residuals = model.resid
    
    norm_cooksd = (cooks_distance - np.min(cooks_distance)) / (np.max(cooks_distance) - np.min(cooks_distance))

    
    residual_df = pd.DataFrame({'Leverage': leverage, 'Standardized residual':residuals, 'Normalized Cook\'s Distance': norm_cooksd})
    scatter = residual_df.hvplot.scatter('Leverage', 'Standardized residual', c='Normalized Cook\'s Distance')
    
    scatter.opts(line_color='black')
    
    return scatter * hv.Slope.from_scatter(scatter).opts(line_color='red',line_width=1)


### Classification Plot

# def plot_roc(classification):
    
#     # Calculer le taux de vrais positifs (true positive rate) et le taux de faux positifs (false positive rate)
#     fpr, tpr, _ = roc_curve(classification.y_test_cl, classification.y_score_cl)
    
#     # Calculer l'aire sous la courbe ROC (AUC)
#     roc_auc = auc(fpr, tpr)
    
#     # Tracer la courbe ROC
#     fig = plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     fig = pn.pane.Matplotlib(fig)
#     fig.sizing_mode = 'scale_both'
#     plt.close()
#     return fig

def plot_roc(classification):
    
    # Calculer le taux de vrais positifs (true positive rate) et le taux de faux positifs (false positive rate)
    fpr, tpr, _ = roc_curve(classification.y_test_cl, classification.y_score_cl)
    
    # Calculer l'aire sous la courbe ROC (AUC)
    roc_auc = auc(fpr, tpr)
    
    # Créer une courbe ROC à l'aide de hvplot
    roc_curve_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    roc_curve_plot = roc_curve_df.hvplot.line(x='FPR', y='TPR', line_color='darkorange', 
                                           line_width=2, title=f"ROC Curve (AUC = {roc_auc:.2f})",
                                           xlim=(0,1), ylim=(0,1))
    roc_curve_plot *= hv.Curve([(0, 0), (1, 1)]).opts(line_color='darkblue')
    roc_curve_plot.opts(xlabel='False Positive Rate', ylabel='True Positive Rate', show_legend=True, legend_position='bottom_right')
    
    return roc_curve_plot

def confusion_matrix_heatmap(classification, color):
    # Compute the confusion matrix
    cm = pd.crosstab(
        classification.y_test_cl, classification.y_pred_cl, normalize='index'
    )

    # Create a Plotly heatmap
    fig = ff.create_annotated_heatmap(
        z=cm.values,
        x=list(cm.columns),
        y=list(cm.index),
        annotation_text=cm.round(2).values,
        colorscale=color,
        zmin=0,
        zmax=1,
        showscale=True
    )

    # Update layout
    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(side='bottom', tickangle=0),
        yaxis=dict(autorange='reversed')
    )

    return fig


## Embedding plot
def plot_digits_embedding(X2d, y, title=None, remove_ticks=True):
  """
  Plot a 2D points at positions `X2d` using text labels from `y`.
  The data is automatically centered and rescaled to [0,1].
  Ticks are removed by default since the axes usually have no meaning (except for PCA).
  """
  x_min, x_max = np.min(X2d, 0), np.max(X2d, 0)
  X = (X2d - x_min) / (x_max - x_min)

  plt.figure(figsize=(20,10))
  ax = plt.subplot(111)
  for i in range(X.shape[0]):
    plt.text(X[i, 0], X[i, 1], str(y[i]),
                color=plt.cm.tab10(y[i]),
                fontdict={'weight': 'bold', 'size': 9})

  if remove_ticks:
    plt.xticks([]), plt.yticks([])
  if title is not None:
    plt.title(title)




################# MODEL

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class History:
    def __init__(self, model, X_train, y_train, X_test, y_test, y_pred, residuals):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.residuals = residuals
        
    def to_dict(self):
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'model': self.model,
            'y_pred': self.y_pred,
            'residuals': self.residuals
        }

from sklearn.model_selection import train_test_split
def model_history(df, target, model):
    
    Y = df[target]

    columns_to_drop = ['pass', 'target_name', 'id', target]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    X = df.drop(columns_to_drop, axis=1)
    
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.3, random_state=123)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_instance = model()
    model_instance.fit(X_train, y_train)

    y_pred = model_instance.predict(X_test)

    residuals = y_test - y_pred

    history = History(str(model_instance)[:-2], X_train, y_train, X_test, y_test, y_pred, residuals)
    return history


def df_reg(df):
    df_copy = df.copy()
    df_copy = pd.get_dummies(df_copy, prefix='gender_', columns=['gender'])
    df_copy = pd.get_dummies(df_copy, prefix='race_', columns=['race/ethnicity'])
    df_copy = pd.get_dummies(df_copy, prefix='lunch_', columns=['lunch'])
    edu_dict = {"some high school": 0, "high school": 1, "some college": 2, "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5}
    df_copy['parental level of education'] = df_copy['parental level of education'].replace(edu_dict)
    edu_dict = {"none": 0, "completed": 1}
    df_copy['test preparation course'] = df_copy['test preparation course'].replace(edu_dict)
    df_copy["average_score"] = df_copy[["math score", "reading score", "writing score"]].mean(axis=1)
    df_copy.drop(['math score', 'reading score', 'writing score','gender__female','lunch__free/reduced','pass','target_name'], axis=1,inplace=True)
    return df_copy
    

# Classification

class Classification:
    def __init__(self, model_cl, X_train_cl, y_train_cl, X_test_cl, y_test_cl, y_pred_cl, y_score_cl,classes):
        self.model_cl = model_cl
        self.X_train_cl = X_train_cl 
        self.y_train_cl = y_train_cl
        self.X_test_cl = X_test_cl
        self.y_test_cl = y_test_cl
        self.y_pred_cl = y_pred_cl
        self.y_score_cl = y_score_cl
        self.classes = classes
        
    def to_dict(self):
        return {
            'X_train': self.X_train_cl,
            'y_train': self.y_train_cl,
            'X_test': self.X_test_cl,
            'y_test': self.y_test_cl,
            'model': self.model_cl,
            'y_pred': self.y_pred_cl,
            'y_score':self.y_score_cl,
            'classes': self.classes
        }
    
def model_cl_history(df, model_cl):

    Y = df['pass']
    
    columns_to_drop = ['pass', 'target_name', 'id','math score','reading score','writing score']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    X = df.drop(columns_to_drop, axis=1)
    
    X = pd.get_dummies(X)

    X_train_cl, X_test_cl, y_train_cl, y_test_cl  = train_test_split(X, Y, test_size=0.15, random_state=42)

    if model_cl == SVC:
        model_instance = model_cl(probability=True)

    else:
        model_instance = model_cl()
        
    model_instance.fit(X_train_cl, y_train_cl)

    y_pred_cl = model_instance.predict(X_test_cl)
    y_score_cl = model_instance.predict_proba(X_test_cl)[:,1]

    classes = np.unique(Y)

    classification = Classification(str(model_instance)[:-2], X_train_cl, y_train_cl, X_test_cl, y_test_cl, y_pred_cl,y_score_cl, classes)
    return classification

######## Dashboard
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

df = pd.read_csv("data\StudentsPerformance.csv")
numeric_features = ['math score', 'reading score', 'writing score']
categoric_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
df['pass'] = df.apply(lambda row: 1 if row['math score'] >= 60 and row['reading score'] >= 60 and row['writing score'] >= 60 else 0, axis=1)


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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




