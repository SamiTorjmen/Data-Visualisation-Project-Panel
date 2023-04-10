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