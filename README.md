# Housing Price Predictor
In this project, I aim to create multiple models for housing price prediction using a dataset from Cook County, Illinois. I will be starting with a more simple linear regression model,
before attempting a random forest model, and finally a neural net. 

## Important Background Information
Cook County, Illinois has been redlined historically, and some of the house price estimates still reflect that. Within our model, we will be doing our best to not include such biases within our predictions.

## Exploratory Data Analysis
This file contains all the exploratory data analysis. The data is read in as a csv, and goes through ETL to determine the useful features. We also plot the distribution of some features to check if there is indeed a linear relationship. The data covered multiple orders of magnitude, so we log transformed the data in order to have a normal distribution with a more linear relationship. These features are what are used to build the models

## Linear Regression Model
This is the first and most basic of the models that we create in this project. We first create a basic model with only two features to test that the data pipeline works as we want it to, and that the model works. Afterwards we create two modesl - one including the Estimate (Land) and Estimate (Building) features, one without. The reason for this is because of the historic redlining. Estimate (Land) and Estimate (Building) were the estimation of the land cost and building cost in the previous year. If the previous year's pricing was biased, I don't want to bring that bias into my model. Thus I made the decision to exclude that feature when creating the final model, as well as in future models. The simplicity of a linear regression model allows it to be a good model for establishing a baseline. It is simple to implement and computationally cheap to provide a baseline performance measure to compare with more complex models.

## Random Forest Model
This is a more complex model and achieves a higher accuracy than a linear model. It performs bootstrapping with our data to create multiple decision trees, and uses an ensemble to predict the housing prices. Again, we exclude Estimate (Land) and Estimate (Building) from our model. A random forest model can capture more complex, non-linear relationships, but requires more compute to create than a linear regression model. It can show some feature importances, but is still generally considered a black box model as the decision making process isn't as straightforward. This means that the model is a bit more difficult to explain.

## Model
This is our final and most complex model. We implement a neural network with the same features used in the random forest model, but achieve a much higher accuray. This is the model that we employ in Website.ipynb. Neural networks require the most compute out of all the implemented models, and due to it's power can often overfit when datasets are too small. This is also a black box model which could be a drawback when transparency is important.

### Error Calculation
In our models we use root mean squared error as our loss function. We chose RMSE over other loss functions such as mean squared error (MSE) or mean absolute error (MAE) because it in in the same metrics as the target variable, making it easier to interpret, and penalizes large errors more which leads to less extreme predictions.

## Helper Functions
This is a python file that contains all the useful functions we use throughout the project.

fill_missing(data, col): Fills in missing values in a column with the median value

rmse(predicted, actual): Calculates the rmse of a set of predictions.

process_data_pipe(data, pipeline_functions, prediction_col): Generates X, Y after processing the data through the pipeline functions paying attention to only the columns specified in prediction_col

select_columns(data, columns): Selects columns from data

train_val_split(data): Splits data into a train and validation set

ohe_roof_material(data): One hot encodes the roof material column

substitute_roof_material(data): Creates columns corresponding to different roof materials

add_bedrooms(data): Adds a new column containing the number of bedrooms in a house

add_total_rooms(data): Adds a new column containing total number of rooms in a house

add_bathrooms(data): Adds a new column containing total number of bathrooms in a house

log_transform(data, column): Adds the log of a column to the passed in dataframe with the name 'Log {column}'

remove_outliers(data, variable, lower, higher): Removes outliers from a variable that are less than lower and greater than higher

plot_distribution(data, label, rows): Plots the distribution of label as a histplot and a boxplot

head(filename, lines): Returns the first lines of filename

fetch_and_cache(data_url, file, data_dir, force): Downloads and caches a url and returns the file object

line_-count(file): Returns the number of linesi n a file

run_linear_regression_test(final_model, process_data_fm, threshold, train_data_path, test_data_path, is_test, is_ranking, return_predictions): Tests whether a linear regression model has a loss lower than a threshold

run_linear_regression_test_optim(final_model, process_data_fm. threshold, train_data_path, test_data_path, is_test, is_ranking, return_predictions): Tests the accuracy of a linear model
