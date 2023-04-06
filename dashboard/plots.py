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

import matplotlib.pyplot as plt
import seaborn as sns

import panel as pn
pn.extension('plotly')

from dashboard.helpers import plotly_to_plt_colors, color_s, categarray


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
    
    return fig

## Heatmap
def corr_heatmap(df, numeric_features,color):

    # Calculate the correlation matrix
    corrmat = df[numeric_features].corr(method='pearson')

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
        residual_df =  pd.DataFrame({'Predicted Values': history.y_pred, 'Root Standardized Residuals': history.residuals.apply(lambda x: np.sqrt(np.abs(x)))})
        scatter = residual_df.hvplot.scatter('Predicted Values', 'Root Standardized Residuals')

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




