# Housing Price Predictor
In this project, I aim to create multiple models for housing price prediction using a dataset from Cook County, Illinois. I will be starting with a more simple linear regression model,
before attempting a random forest model, and finally a neural net. 

## Important Background Information
Cook County, Illinois has been redlined historically, and some of the house price estimates still reflect that. Within our model, we will be doing our best to not include such biases within our predictions.

## Exploratory Data Analysis
This file contains all the exploratory data analysis. The data is read in as a csv, and goes through ETL to determine the useful features. These features are what are used to build the models

## Linear Regression Model
This is the first and most basic of the models that we create in this project. We first create a basic model with only two features to test that the data pipeline works as we want it to, and that the model works. Afterwards we create two modesl - one including the Estimate (Land) and Estimate (Building) features, one without. The reason for this is because of the historic redlining. Estimate (Land) and Estimate (Building) were the estimation of the land cost and building cost in the previous year. If the previous year's pricing was biased, I don't want to bring that bias into my model. Thus I made the decision to exclude that feature when creating the final model, as well as in future models.

## Random Forest Model
This is a more complex model and achieves a higher accuracy than a linear model. It performs bootstrapping with our data to create multiple decision trees, and uses an ensemble to predict the housing prices. Again, we exclude Estimate (Land) and Estimate (Building) from our model.

## Model
This is our final and most complex model. We implement a neural network with the same features used in the random forest model, but achieve a much higher accuray. This is the model that we employ in Website.ipynb. 

## Website
We use voila and ipywidgets in order to create a website where you can enter the features of a house and have the model predict the price your house could be.

