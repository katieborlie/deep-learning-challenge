# Module 21 Challenge: deep-learning-challenge 🔠 

#### Google Colab Links:
#### #1: [Starter_Code.ipynb](https://colab.research.google.com/drive/1ak_AR3BpZQPheIT4QUpmHujZgXfzzvT4?usp=sharing)

#### #2: [AlphabetSoupCharity_Optimization.ipynb](https://colab.research.google.com/drive/1FGHRQKViuSwjv4Bt9QKWXPX4eKoeeera?usp=sharing) 

<img width="100%" alt="Screenshot 2023-05-10 at 4 19 15 PM" src="https://github.com/katieborlie/deep-learning-challenge/assets/119274891/2afa0c68-1517-48ce-8b9a-a9b1335c236e">

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

  #* **Data Preprocessing**

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

   * I've decided that variables `EIN` and `NAME` should be removed from the input data becuase they are neither targets nor relevant features of our analysis.


<img width="100%" alt="Screenshot 2023-05-24 at 1 18 21 AM" src="https://github.com/katieborlie/deep-learning-challenge/assets/119274891/28a37e77-277d-4af7-852c-7458d711a82c">


  #* **Compiling, Training, and Evaluating the Model**

   1. How many neurons, layers, and activation functions did you select for your neural network model, and why?

   * For my neural network model, I selected three hidden layers with neuron amounts of 20, 26, and 3 because this grouping seemed to be pretty accurate in determining loss (I tried running a few different iterations with varying combinations). The activation function I chose for the first two hidden layers were ReLU, in order to improve performance and explore non-linearity. For the third hidden layer, I used sigmoid as my activation function since this is a binary classification task and sigmoid is the most convenient for this case. Lastly, for the output layer, I used sigmoid activation because I am dealing with predictability and need to be sure that my output is between 0 and 1.


<img width="100%" alt="Screenshot 2023-05-24 at 1 43 01 AM" src="https://github.com/katieborlie/deep-learning-challenge/assets/119274891/e0758561-d974-48f7-9282-4b8ff5817d16">


   2. Were you able to achieve the target model performance?

   * According to my analysis, I was able to achieve an accuracy score of 73%. While I would've liked to see numbers above 75%, I think this is still a useful deep neural network model for Alphabet Soup to utilize when selecting applicants with the best chances of success.


<img width="1025" alt="Screenshot 2023-05-24 at 1 44 21 AM" src="https://github.com/katieborlie/deep-learning-challenge/assets/119274891/9fc80491-66dc-4e31-ac45-e990b8e3f790">


   3. What steps did you take in your attempts to increase model performance?

   * To increase model performance, I began by dropping the `EIN` and `NAME` variables to eliminate any variables that could decrease my accuracy. I also chose a cutoff value and created a list of application types to be replaced (`application_types_to_replace`) using `value_counts` so that any application type that had less than 500 occurrences was replaced with "Other". Likewise, a cutoff value was chosen for the `CLASSIFICATION` column and any classification with fewer than 1000 occurrences was replaced with "Other". The binning was checked to ensure that it was successful.

   
<img width="1010" alt="Screenshot 2023-05-24 at 1 54 40 AM" src="https://github.com/katieborlie/deep-learning-challenge/assets/119274891/46b6f801-2a2e-492f-86a2-2dc4ae2eeec7">

<img width="1011" alt="Screenshot 2023-05-24 at 1 59 29 AM" src="https://github.com/katieborlie/deep-learning-challenge/assets/119274891/ecd112de-c740-48cf-b9ab-9b7499ae48c9">


#### Summary: (Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.)
   
   * Overall, my deep learning model using TensorFlow and Keras was able to achieve a predictive accuracy of 73% in classifying the success of organizations funded by Alphabet Soup based on their features. The model underwent several optimization attempts, including dropping columns, binning categorical variables, adding hidden layers and neurons, and trying different activation functions, among other adjustments. 
     If I were to recommend a different model to solve this classification problem, I would suggest a Support Vector Machine (SVM) which deals with numerical and categorical variables as well as imbalanced datasets. These types of models are used for solving binary classification problems and could potentially increase the accuracy without resorting to various optimization techniques. It is always worth comparing models to get the highest level of accuracy.
