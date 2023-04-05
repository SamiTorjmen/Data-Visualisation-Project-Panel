import pandas as pd
import panel as pn
pn.extension('plotly')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


pn.config.sizing_mode = "stretch_width"


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
    
    return pn.widgets.Tabulator(df_g, page_size=10,sizing_mode='stretch_both')


def filtered_dataframe(df, **checkboxes_values):
    '''
    A reactive function to filter the dataframe based on the checked checkboxes
    ---------------------------------------------------------------------------
    df -> DataFrame
    checkboxes_values -> Dict 
    '''
    selected_columns = [col for col, value in checkboxes_values.items() if value]
    return pn.widgets.Tabulator(df[selected_columns], page_size=10)



def evaluate_regression_model(history):

    # Calculate metrics
    mse = mean_squared_error(history.y_test, history.y_pred)
    rmse = mean_squared_error(history.y_test, history.y_pred, squared=False)
    mae = mean_absolute_error(history.y_test, history.y_pred)
    r2 = r2_score(history.y_test, history.y_pred)
    # Create a dictionary with the results
    results = {'R2 Score': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    # Create a DataFrame from the dictionary and return it
    eval_df = pd.DataFrame.from_dict(results, orient='index', columns=[str(history.model)])
    return pn.widgets.Tabulator(eval_df, page_size=10)