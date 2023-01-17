# Module 21 Challenge

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using machine learning and neural networks, a binary classifier was created on the provided dataset that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, a CSV was provided containing more than 34,000 organisations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organisation, such as:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organisation classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organisation type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively


## Instructions

### Step 1: Preprocess the Data
Using Pandas and scikit-learn’s StandardScaler(), the dataset was preprocessed. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

1. Read in the charity_data.csv to a Pandas DataFrame, and the following were identified in the dataset:
  - What variable(s) are the target(s) for your model?
  - What variable(s) are the feature(s) for your model?
  - Drop the EIN and NAME columns.

2. Determine the number of unique values for each column.
  - For columns that have more than 10 unique values, the number of data points for each unique value was determined.
  - Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

3. Use pd.get_dummies() to encode categorical variables.

4. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

5. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, a neural network or deep learning modelw as designed to create a binary classification model that can predict if an Alphabet Soup-funded organisation will be successful based on the features in the dataset. The number of inputs were considered before determining the number of neurons and layers in the model. Once this step was completed, compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.

Using the same Jupyter Notebook in which the preprocessing steps from Step 1 was performed,

  - Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

  - Create the first hidden layer and choose an appropriate activation function. If necessary, add a second hidden layer with an appropriate activation function.

  - Create an output layer with an appropriate activation function.

  - Check the structure of the model.

  - Compile and train the model.

  - Create a callback that saves the model's weights every five epochs.

  - Evaluate the model using the test data to determine the loss and accuracy.

  - Save and export the results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### Step 3: Optimise the Model

Using TensorFlow, optimise the model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimise the model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  - Dropping more or fewer columns.
  - Creating more bins for rare occurrences in columns.
  - Increasing or decreasing the number of values for each bin.

Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.

A new Jupyter Notebook file was created and named AlphabetSoupCharity_Optimisation#.ipynb, with the "#" representing the file optimisation number.

The preprocessing steps were repeated as in Step 1. The number of neurons in the hidden layers were modifed as well as the number of hidden layers with the aim of achieving higher than 75% accuracy. On the 3rd optimisation, the activation function of the input layer was changed from Relu to Tanh. 

The results were saved to an HDF5 file with the filename AlphabetSoupCharity_Optimisation#.h5.

### Step 4: Report on the Neural Network Model

A report was prepared on the performance of the deep learning model created for Alphabet Soup.

The report contained the following:

Overview of the analysis explaining the purpose of this analysis.

Results summarising the following:

- Data Preprocessing

- Compiling, Training, and Evaluating the Model

- Summary
