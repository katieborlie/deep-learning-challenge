# Module 21 Challenge: deep-learning-challenge 🔠 

#### Google Colab Links:
#### #1: [Starter_Code.ipynb](https://colab.research.google.com/drive/1ak_AR3BpZQPheIT4QUpmHujZgXfzzvT4?usp=sharing)

#### #2: [AlphabetSoupCharity_Optimization.ipynb](https://colab.research.google.com/drive/1FGHRQKViuSwjv4Bt9QKWXPX4eKoeeera?usp=sharing) 

<img width="100%" alt="Screenshot 2023-05-10 at 4 19 15 PM" src="https://github.com/katieborlie/deep-learning-challenge/assets/119274891/2afa0c68-1517-48ce-8b9a-a9b1335c236e">

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

  * **EIN** and **NAME**—Identification columns

  * **APPLICATION_TYPE**—Alphabet Soup application type
 
  * **AFFILIATION**—Affiliated sector of industry

  * **CLASSIFICATION**—Government organization classification

  * **USE_CASE**—Use case for funding

  * **ORGANIZATION**—Organization type

  * **STATUS**—Active status

  * **INCOME_AMT**—Income classification

  * **SPECIAL_CONSIDERATIONS**—Special considerations for application

  * **ASK_AMT**—Funding amount requested

  * **IS_SUCCESSFUL**—Was the money used effectively

## Instructions

### Step 1: Preprocess the Data

Using my knowledge of Pandas and scikit-learn’s `StandardScaler()`, I’ll need to preprocess the dataset. This step prepares me for Step 2, where I'll compile, train, and evaluate the neural network model.

I'll start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:

  * What variable(s) are the target(s) for your model?

  * What variable(s) are the feature(s) for your model?

2. Drop the `EIN` and `NAME` columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

6. Use `pd.get_dummies()` to encode categorical variables.

7. Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

### Step 2: Compile, Train, and Evaluate the Model

Using my knowledge of TensorFlow, I’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. I’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once I’ve completed that step, I’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1, Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

Using my knowledge of TensorFlow, I'll optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

  * Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:

    * Dropping more or fewer columns.

    * Creating more bins for rare occurrences in columns.

    * Increasing or decreasing the number of values for each bin.

    * Add more neurons to a hidden layer.

    * Add more hidden layers.

    * Use different activation functions for the hidden layers.

    * Add or reduce the number of epochs to the training regimen.

> **_NOTE:_** If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

### Step 4: Written Report on the Neural Network Model

#### Overview of the analysis: 

The purpose of this analysis is to help the nonprofit foundation Alphabet Soup to select applicants for funding with the best chance of success. I’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

  * **EIN** and **NAME**—Identification columns

  * **APPLICATION_TYPE**—Alphabet Soup application type
 
  * **AFFILIATION**—Affiliated sector of industry

  * **CLASSIFICATION**—Government organization classification

  * **USE_CASE**—Use case for funding

  * **ORGANIZATION**—Organization type

  * **STATUS**—Active status

  * **INCOME_AMT**—Income classification

  * **SPECIAL_CONSIDERATIONS**—Special considerations for application

  * **ASK_AMT**—Funding amount requested

  * **IS_SUCCESSFUL**—Was the money used effectively

#### Results: 

  * **Data Preprocessing**

    1. What variable(s) are the target(s) for your model?

    * The target variable for my model is `IS_SUCCESSFUL` since we are trying to select applicants with the best chance of success. This variable is a classification of the binary outcome variable regarding success in charity donations/if money was used effectively.

    2. What variable(s) are the features for your model?

    * The variables that are features in my model are all of the other columns besides `IS_SUCCESSFUL`, including:

     * **EIN** and **NAME**—Identification columns

     * **APPLICATION_TYPE**—Alphabet Soup application type
 
     * **AFFILIATION**—Affiliated sector of industry

     * **CLASSIFICATION**—Government organization classification

     * **USE_CASE**—Use case for funding

     * **ORGANIZATION**—Organization type
  
     * **STATUS**—Active status

     * **INCOME_AMT**—Income classification

     * **SPECIAL_CONSIDERATIONS**—Special considerations for application

     * **ASK_AMT**—Funding amount requested

    3. What variable(s) should be removed from the input data because they are neither targets nor features?

    * 

  * **Compiling, Training, and Evaluating the Model**

    * How many neurons, layers, and activation functions did you select for your neural network model, and why?

    * Were you able to achieve the target model performance?

    * What steps did you take in your attempts to increase model performance?

#### Summary: (Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.)
    + insert
