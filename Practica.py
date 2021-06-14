import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale

# Long output for DataFrame.head()
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None)



# ================= F U N C T I O N S =====================

def get_data(file_name, sim_or_exp, PDD = None):
    """ Get data from file_name

    Parameters
    ----------
    file_name: str_like
        Name of the file to be read

    sim_or_exp: str_like
        "sim" for simulated data or "exp" for experimental data

    PDD: bool_like
        True to import PDD data. Default is None

    Returns
    -------
    DataFrame with all the data
    """
    if sim_or_exp == "sim": 
        if PDD: 
            data = pd.read_fwf(file_name + ".plotdat", header=None).dropna()[[0, 1]]
            data[0] = data[0] * 10
        else:
            data = pd.read_fwf(file_name + ".egslst", header=None).dropna()
            data[0] = data[0] * 10
        return data
    elif sim_or_exp == "exp": 
        path_to_exp = "./Material_laboratorio/"
        return pd.read_csv(path_to_exp + file_name + ".dat", delimiter="\t", header=None).dropna()[[0, 1]]
    else: raise ValueError("[ERROR] Only values for sim_or_exp are sim or exp")


def normalizer(data, PDD = None):
    """ Normalize the dose to the max value

    Parameters
    ----------
    data: DataFrame_like

    Returns
    -------
    DataFrame with normalized dose
    """
    if PDD:
        data[1] = [x/np.abs(max(data[1])) for x in data[1]]
    else:
        data[1] = minmax_scale(data[1])
    return data


def plotter(repr_type):
    """ Plot the desired representation

    Parameters
    ----------
    repr_type: str_like
        Representation type. Can be x, y or PDD.

    Returns
    -------
    Representation plot
    """

    repr_type = repr_type.upper()
    resolution = [2560, 1600]
    plt.figure(figsize = (10 * resolution[0] / resolution[1] , 10))
    plt.xlabel("Distance [mm]", fontsize = 28)
    plt.ylabel("Dose [Gy]", fontsize = 28)

    if repr_type == "X" or repr_type == "Y":
        sim_values = globals()["prof_data" + repr_type]
        exp_values = globals()["prof_data_exp" + repr_type]
        plt.plot(sim_values[0], 1-sim_values[1], label = "Simulated")
        plt.plot(exp_values[0], 1-exp_values[1], label="Experimental")
        plt.title("Profile in " + repr_type, fontsize=40)
        plt.legend(fontsize=14)
        plt.savefig("Profile_in_" + repr_type + ".jpg", dpi = 300, bbox_inches="tight")

    elif repr_type == "PDD":
        plt.plot(PDD_data[0], PDD_data[1], label = "Simulated")
        plt.plot(PDD_data_exp[0], PDD_data_exp[1], label="Experimental")
        plt.title(repr_type, fontsize=40)
        plt.legend(fontsize=14)
        plt.savefig(repr_type + ".jpg", dpi = 300, bbox_inches="tight")




# ============= I M P O R T   T H E   D A T A =====================

# Import data in X
prof_dataX = -get_data("profile_X", "sim")
prof_data_expX = get_data("profile_X", "exp")

# Import data in Y
prof_dataY = get_data("profile_Y", "sim")
prof_data_expY = get_data("profile_Y", "exp")
prof_dataY[1] = -prof_dataY[1]

# Import PDD data
PDD_data = get_data("PDD3", "sim", PDD = True)
PDD_data_exp = get_data("PDD", "exp")



# =============== N O R M A L I Z A T I O N ======================

# Apply the normalizer function to all the datas
datas = [prof_dataX, prof_dataY, prof_data_expX, prof_data_expY, PDD_data, PDD_data_exp]
[*datas] = map(normalizer, datas, [True if i >= len(datas) - 2 else False for i in range(len(datas))])



# ======================= P L O T S =============================

plotter("X"), plotter("Y"), plotter("PDD")
plt.show()
