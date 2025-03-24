"""
Helper Functions
"""
import pandas as pd
import numpy as np
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K

def nn_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def fill_missing(data, col):
    copy = data.copy()
    copy[col] = copy[col].fillna(np.median(copy[col]))
    return copy

def rmse(predicted, actual):
    return np.sqrt(np.mean((actual - predicted)**2))


def process_data_pipe(data, pipeline_functions, prediction_col):
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    X = data.drop(columns = [prediction_col])
    Y = data.loc[:, prediction_col]
    return X, Y


def select_columns(data, *columns):
    return data.loc[:, columns]

def train_val_split(data):
    data_len = data.shape[0]
    shuffled_indices = np.random.permutation(data_len)
    train_indices = shuffled_indices[:int(data_len * 0.8)]
    validation_indices = shuffled_indices[int(data_len * 0.8):]
    train = data.iloc[train_indices]
    validation = data.iloc[validation_indices]
    return train, validation

def ohe_roof_material(data):
    cat = ['Roof Material']
    oh_enc = OneHotEncoder()
    oh_enc.fit(data[cat])

    cat_data = oh_enc.transform(data[cat]).toarray()
    cat_df = pd.DataFrame(data = cat_data, columns = oh_enc.get_feature_names_out(), index = data.index)
    return data.join(cat_df)

def substitute_roof_material(data):
    copy = data.copy()
    copy["Roof Material"] = copy['Roof Material'].map({1:"Shingle/Asphalt", 2:"Tar & Gravel", 3:"Slate", 4: "Shake", 5: "Tile", 6: "Other"})
    return copy

def add_bedrooms(data):
    with_rooms = data.copy()
    with_rooms['Bedrooms'] = with_rooms['Description'].str.extract(r'(\d+) of which are bedrooms,').astype(int).fillna(0)
    return with_rooms

def add_total_rooms(data):
    with_rooms = data.copy()
    with_rooms['Rooms'] = with_rooms['Description'].str.extract(r'It has a total of (\d+) rooms,').astype(int).fillna(0)
    return with_rooms

def add_bathrooms(data):
    with_rooms = data.copy()
    with_rooms['Bathrooms'] = with_rooms['Description'].str.extract(r'(\d+(?:\.\d+)?) of which are bathrooms.').astype(float).fillna(0)
    return with_rooms

def log_transform(data, col):
    copy = data.copy()
    copy['Log ' + col] = np.log(data[col] + 1)
    return copy

def remove_outliers(data, variable, lower = -np.inf, upper = np.inf):
    # Remove outliers from data in column variable that are lower than lower and higher than upper
    data = data[(data[variable] > lower) & (data[variable] <= upper)]
    return data

def plot_distribution(data, label, rows):
    fig, axs = plt.subplots(nrows = rows)

    sns.histplot(data[label], kde=True, ax=axs[0], stat = 'density', bins = 100)
    sns.boxplot(x=data[label], width = 0.3, ax = axs[1], showfliers = False)

    spacer = np.max(data[label]) * 0.05
    xmin = np.min(data[label]) - spacer
    xmax = np.max(data[label]) + spacer
    axs[0].set_xlim((xmin, xmax))
    axs[1].set_xlim((xmin, xmax))
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)

    plt.subplots_adjust(hspace = 0)
    fig.suptitle("Distribution of " + label)

def head(filename, lines=5):
    """
    Returns the first few lines of a file.
    
    filename: the name of the file to open
    lines: the number of lines to include
    
    return: A list of the first few lines from the file.
    """
    from itertools import islice
    with open(filename, "r") as f:
        return list(islice(f, lines))
    

def fetch_and_cache(data_url, file, data_dir="data", force=False):
    """
    Download and cache a url and return the file object.
    
    data_url: the web address to download
    file: the file in which to save the results.
    data_dir: (default="data") the location to save the data
    force: if true the file is always re-downloaded
    
    return: The pathlib.Path object representing the file.
    """

    import requests
    from hashlib import md5
    from pathlib import Path
    
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir/Path(file)
    # If the file already exists and we want to force a download then
    # delete the file first so that the creation date is correct.
    if force and file_path.exists():
        file_path.unlink()
    if force or not file_path.exists():
        resp = requests.get(data_url, stream=True)
        file_size = int(resp.headers.get('content-length', 0))
        step = 40
        chunk_size = file_size//step
        with file_path.open('wb') as f:
            for chunk in resp.iter_content(chunk_size): # write file in chunks
                f.write(chunk)
                step -= 1
                print('[' + '#'*(41 - step) + (step)*' ' + ']\r', end='')
        print(f"\nDownloaded {data_url.split('/')[-1]}!")
    else:
        import time
        time_downloaded = time.ctime(file_path.stat().st_ctime)
        print("Using version already downloaded:", time_downloaded)
    # Compute and print md5 hash of file, whether newly downloaded or not
    m5 = md5()
    m5.update(file_path.read_bytes())
    print(f"MD5 hash of file: {m5.hexdigest()}")
    return file_path


def line_count(file):
    """
    Computes the number of lines in a file.
    
    file: the file in which to count the lines.
    return: The number of lines in the file
    """
    with open(file, "r") as f:
        return sum(1 for line in f)

def run_linear_regression_test(
    final_model, 
    process_data_fm, 
    threshold, 
    train_data_path, 
    test_data_path, 
    is_test=False, 
    is_ranking=False,
    return_predictions=False
):
    def rmse(predicted, actual):
        return np.sqrt(np.mean((actual - predicted)**2))

    training_data = pd.read_csv(train_data_path, index_col='Unnamed: 0')
    X_train, y_train = process_data_fm(training_data)
    if is_test:
        test_data = pd.read_csv(test_data_path, index_col='Unnamed: 0')
        X_test = process_data_fm(test_data, is_test_set = True)
        assert len(test_data) == len(X_test), 'You may not remove data points from the test set!'

    final_model.fit(X_train, y_train)
    if is_test:
        return final_model.predict(X_test)
    else:
        y_predicted = final_model.predict(X_train)
        loss = rmse(np.exp(y_predicted), np.exp(y_train))
        if is_ranking:
            print('Your RMSE loss is: {}'.format(loss))
            return loss
        return loss < threshold
    
def run_linear_regression_test_optim(
    final_model, 
    process_data_fm, 
    train_data_path, 
    test_data_path, 
    is_test=False, 
    is_ranking=False,
    return_predictions=False
):
    def rmse(predicted, actual):
        return np.sqrt(np.mean((actual - predicted)**2))

    training_data = pd.read_csv(train_data_path, index_col='Unnamed: 0')
    X_train, y_train = process_data_fm(training_data)
    if is_test:
        test_data = pd.read_csv(test_data_path, index_col='Unnamed: 0')
        X_test = process_data_fm(test_data, is_test_set = True)
        assert len(test_data) == len(X_test), 'You may not remove data points from the test set!'

    final_model.fit(X_train, y_train)
    if is_test:
        return final_model.predict(X_test)
    else:
        y_predicted = final_model.predict(X_train)
        loss = rmse(np.exp(y_predicted), np.exp(y_train))
        if is_ranking:
            print('Your RMSE loss is: {}'.format(loss))
            return loss
        fn = (lambda threshold: loss < threshold)
        fn.loss = loss
        fn.signature = (process_data_fm, train_data_path, test_data_path)
        return fn