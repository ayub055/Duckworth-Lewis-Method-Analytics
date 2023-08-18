import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


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
        pass

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
    data = data.dropna()
    data = select_columns(data, columns_to_keep)
    print("\nDetails After Preprocessing\n-------------\n")
    null_details(data)
    
    return data


def train_model(data: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    """Trains the model

    Args:
        data (pd.DataFrame, np.ndarray): Datastructure containg the cleaned data
        model (DLModel): Model to be trained
    """
    
    return model


def plot(model: DLModel, plot_path: str) -> None:
    """ Plots the model predictions against the number of overs
        remaining according to wickets in hand.

    Args:
        model (DLModel): Trained model
        plot_path (str): Path to save the plot
    """
    pass


def print_model_params(model: DLModel) -> List[float]:
    '''
    Prints the 11 (Z_0(1), ..., Z_0(10), L) model parameters

    Args:
        model (DLModel): Trained model
    
    Returns:
        array: 11 model parameters (Z_0(1), ..., Z_0(10), L)

    '''
    
    return []


def calculate_loss(model: DLModel, data: Union[pd.DataFrame, np.ndarray]) -> float:
    '''
    Calculates the normalised squared error loss for the given model and data

    Args:
        model (DLModel): Trained model
        data (pd.DataFrame or np.ndarray): Data to calculate the loss on
    
    Returns:
        float: Normalised squared error loss for the given model and data
    '''
    
    return 0.0


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
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data)


if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path": "../models/model.pkl",  # ensure that the path exists
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)
