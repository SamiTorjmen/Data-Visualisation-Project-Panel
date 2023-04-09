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

