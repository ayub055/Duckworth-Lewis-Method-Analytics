import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
import math



if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = None
        self.loss = 0
    
    def get_predictions(self, X, Z_0=None, w=10, L=None) -> np.ndarray:
        """Get the predictions for the given X values.

        Args:
            X (np.array): Array of overs remaining values.
            Z_0 (float, optional): Z_0 as defined in the assignment.
                                   Defaults to None.
            w (int, optional): Wickets in hand.
                               Defaults to 10.
            L (float, optional): L as defined in the assignment.
                                 Defaults to None.

        Returns:
            np.array: Predicted score possible
        """
        if Z_0 is None:
            Z_0 = self.Z0[w - 1]  # Using Z0 for given wickets
            
        if L is None:
            L = self.L
        
        predicted_scores = Z_0 * (1 - np.exp(-1 * L * X / Z_0))
        return predicted_scores
        

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        """ Calculate the loss for the given parameters and datapoints.
        Args:
            Params (list): List of parameters to be optimized.
            X (np.array): Array of overs remaining values.
            Y (np.array): Array of actual average score values.
            w (int, optional): Wickets in hand.
                               Defaults to 10.

        Returns:
            float: Mean Squared Error Loss for the model parameters 
                   over the given datapoints.
        """
        pass
    
    def save(self, path):
        """Save the model to the given path.

        Args:
            path (str): Location to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """Load the model from the given path.

        Args:
            path (str): Location to load the model.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)
        
        return self


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    """
    Loads the data from the given path and returns a pandas dataframe.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame, np.ndarray: Data Structure containing the loaded data
    """
    try:
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError:
        print(f"File not found at '{data_path}'. Please provide the correct path.")
        return None
    
def null_details(data: Union[pd.DataFrame, np.ndarray]):
    print(f"Are there any Null Values in dataset? : {data.isnull().values.any()}")
    print(f"Total null values in dataset? : {data.isnull().sum().sum()}")
    print(f"Number of rows in our dataset : {len(data)}")
    print(f"Number of columns in our dataset : {len(data.columns)}\n---------------------------\n")
    

def select_columns(data: Union[pd.DataFrame, np.ndarray], columns_to_keep):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    data = data[columns_to_keep]
    return data

def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Preprocesses the dataframe by
    (i)   removing the unnecessary columns,
    (ii)  loading date in proper format DD-MM-YYYY,
    (iii) removing the rows with missing values,
    (iv)  anything else you feel is required for training your model.

    Args:
        data (pd.DataFrame, nd.ndarray): Pandas dataframe containing the loaded data

    Returns:
        pd.DataFrame, np.ndarray: Datastructure containing the cleaned data.
    """
    
    # Dataset has 1109 NAN values in Run.Rate.Required Column Only
    columns_to_keep = ['Innings', 'Innings.Total.Runs', 'Total.Overs', 'Wickets.in.Hand', 'Over', 'Total.Runs']
    
    print("\nDetails before Preprocessing\n-------------\n")
    null_details(data)
    #data = data.dropna()
    data = select_columns(data, columns_to_keep)
    print("\nDetails After Preprocessing\n-------------\n")
    null_details(data)
    
    return data


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel)-> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    def fit_parameters(data):
        '''
        This procedure will fit the curve to optimise the overall loss function against 11 parameters.
        :param innings_number:
        :param runs_scored:
        :param remaining_overs:
        :param wickets_in_hand:
        :return:optimised_res['fun']:Total Loss incurred
        :return:optimised_res['x']:Optimised values of all 11 parameters.
        '''
        parameters = [10, 30, 40, 60, 90, 125, 150, 170, 190, 200,10]

        innings_number    = data['Innings'].values
        remaining_runs    = data['Innings.Total.Runs'].values - data['Total.Runs'].values
        remaining_overs   = data['Total.Overs'].values - data['Over'].values
        wickets_in_hand   = data['Wickets.in.Hand'].values

        # print(f"are lenght of over remaining values equal to wicket in hand : {len(remaining_overs) == len(wickets_in_hand)}")
        optimised_res = sp.optimize.minimize(sum_of_squared_errors_loss_function,parameters,
                          args=[innings_number, remaining_runs, remaining_overs, wickets_in_hand],
                          method='powell')
        return optimised_res['fun'],optimised_res['x']

    def sum_of_squared_errors_loss_function(parameters,args):
        '''
        This procedure defines the objective function which I have passed in scipy.optimize.minimize() function.
        It calculated all total squared error loss for all the data points for innings 1.
        :param parameters: List contains 11 parameters
        :param args: List contains innings_number,runs_scored,remaining_overs,wickets_in_hand
        :return:total_squared_error of the objective function.
        '''
        total_squared_error=0
        l_param=parameters[10]
        innings_number = args[0]
        runs_scored=args[1]
        remaining_overs=args[2]
        wickets_in_hand=args[3]
        for i in range(len(wickets_in_hand)):
            if innings_number[i] == 1:
                runscored = runs_scored[i]
                overremain = remaining_overs[i]
                wicketinhand = wickets_in_hand[i]
                Z0=parameters[wicketinhand - 1]
                if runscored > 0:
                    predicted_run =  Z0 * (1 - np.exp(-1*l_param * overremain / Z0))
                    total_squared_error=total_squared_error + (math.pow(predicted_run - runscored, 2))
        return total_squared_error
    
    
    loss_value, param = fit_parameters(data)
    
    model.L = param[-1]  # Update L value
    model.Z0 = param[:-1]  # Update Z0 valuues
    model.loss = loss_value



    print("Total Loss:", loss_value)
    
    return model



def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    Z0 = model.Z0
    L = model.L
    optparameters = np.insert(Z0, 10, L)
    #print(len(optparameters))
    plt.figure(1)
    plt.title("Expected Runs vs Overs Remaininng")
    plt.xlim((0, 50))
    plt.ylim((0, 250))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 50, 100, 150, 200, 250])
    plt.xlabel('Overs remaining')
    plt.ylabel('Expected Runs')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#555b65', '#999e45', '#222a55']
    x=np.zeros((51))
    for i in range(51):
        x[i]=i
    for i in range(len(optparameters)-1):
        y_run=optparameters[i] * (1 - np.exp(-optparameters[10] * x /optparameters[i]))
        plt.plot(x, y_run, c=colors[i], label='Z[' + str(i + 1) + ']')
        plt.legend()
    plt.savefig(plot_path)
    plt.show()
    plt.close()

def plot_resource_remaining(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    
    Z0 = model.Z0
    L = model.L
    optparameters = np.insert(Z0, 10, L)
    plt.figure(1)
    plt.title("Resource Remaining vs Overs Remaininng")
    plt.xlim((0, 50))
    plt.ylim((0, 100))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Overs remaining')
    plt.ylabel('percentage Of Resource Remaining')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#555b65', '#999e45', '#222a55']
    x = np.zeros((51))
    for i in range(51):
        x[i] = i
    Z5010=optparameters[9] * (1 - np.exp(-optparameters[10] * 50 /optparameters[9]))
    for i in range(len(optparameters)-1):
        y_run=optparameters[i] * (1 - np.exp(-optparameters[10] * x /optparameters[i]))
        plt.plot(x, (y_run/Z5010)*100, c=colors[i], label='Z[' + str(i + 1) + ']')
        plt.legend()
    plt.savefig(plot_path)
    plt.show()
    plt.close()


def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    Z0 = model.Z0
    L = model.L
    param = np.insert(Z0, 10, L)
    
    p_list = []
    for i in range(len(param)):
        p_list.append(param[i])
    
    for i in range(len(param)):
        if(i == 10):
            print("L :"+str(param[i]))
        else:
            print("Z["+str(i+1)+"] :"+str(param[i]))
    return p_list


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    num_data_first_innings = len(data[data['Innings'] == 1])
    loss = model.loss
    normalised_error = loss/num_data_first_innings
    print(normalised_error)
    return normalised_error


def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("\nData loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.\n")
    
    model = DLModel()  # Initializing the model
    model = train_model(data, model)  # Training the model

    model.save(args['model_path'])  # Saving the model

    plot(model, args['plot_path'])  # Plotting the model
    
    #########+++++++++++++++++++++++++++++++++++++++++++++++#########   
    #########   Added for plotting resource remain plot     ######### 
    ########+++++++++++++++++++++++++++++++++++++++++++++++######### 
    
    #plot_resource_remaining(model, '/data/home/ayyoobmohd/Duckworth-Lewis-Stern-Method/plots/plot_resourceremain_vs_overremain.png')

    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot_powell.png",  # ensure that the path exists
    }
    main(args)
