Skincare Product Recommendation Prediction — End-to-End ML Project
Problem Statement:

Can we predict whether a given skincare/makeup product will be well-received (recommended/highly rated) 
by someone with a specific skin type — using only the product's own attributes and the customer's stated skin type 

given a skincare product's price, brand, and a customer's skin type/tone — will that customer recommend the product (is_recommended = 1) or not (= 0)?

This is a binary classification problem. In practical terms, this is the kind of model that could power something like: "Customers with oily skin tend to recommend Product X 85% of the time" or a recommendation engine that flags which products are likely to satisfy a specific type of customer before they even buy it — based on patterns in what similar customers (same skin type, similar price sensitivity, etc.) said in the past.

Dataset

The dataset contains product and customer-level attributes such as:

- Product and brand information
- Price and ratings
- Skin type
- Skin tone
- Product characteristics
- Recommendation/feedback indicators

Target: Product recommendation / positive reception.

This project predicts historical recommendation behavior; it does not determine medical or clinical suitability.

Models Evaluated

Eight classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Neighbors
- XGBoost
- CatBoost

Model Performance


Logistic Regression 63.08%
Decision Tree 63.08%
| Random Forest 63.08%  
| Gradient Boosting 63.08%
| XGBoost 63.08%
| CatBoost 63.08%
| K-Neighbors 61.50%

Best Model: Logistic Regression


Accuracy  63.08%
Precision 63.08%
Recall 100.00%
F1 Score 77.36%
ROC-AUC 55.24%


Key Findings

- The models achieved approximately 63% accuracy.
- Logistic Regression achieved 100% recall, capturing all positive cases in the test set.
- The 77.36% F1 score reflects the balance between precision and recall.
- The 55.24% ROC-AUC indicates limited discriminatory power, suggesting that the available features provide only modest predictive signal.
- Multiple models produced nearly identical accuracy, indicating that additional or better features may be required for meaningful performance improvement.

ML / MLOps Pipeline





Data
  ↓
Data Ingestion
  ↓
Data Transformation
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Hyperparameter Tuning
  ↓
Model Evaluation
  ↓
MLflow Experiment Tracking
  ↓
DagsHub Model Registry
  ↓
Saved Model
  ↓
Prediction Application

Steps:

To run a file with name file_name - type in terminal >> python file_name.py
To add a file to the github repo, use the commands >> git add file_name, then >> git commit -m, then >> git push origin main


Step 0 > Github and evniornment
 Set up a github repository on github where all the code from VS will be committed.
 Then set up an evironment in VSCode using the command >> conda create -p newenvt python== 4.1.1
 Then activate the enviorment using >> conda activate -newenvt
 Then create a readme file here
 Initialise git using >> git init 
 To add this readme file to gitbub repo, run the command >> git add readme.md, then >> git commit -m, then >> git push origin main

Step 1 > Create the gitingnore file and the requirements.txt file
Step 2 > Create an empty fiolder src and then the setup.py file in it along with init.py
Step 3 > Upon running the setup.py file, the ML_Project_Hardika.egg.info is created along with the build and dist files too.
Step 4 > Then a templayte.py file is created. This will define the structure of the project and will create the necessary folders and files for the project. Upon running the template.py file, it will crate all your files memtioned.
Step 5 > Create the logger.py file in the src folder so that instead of using scattered print() statements, your entire project writes timestamped, leveled messages (INFO, ERROR, WARNING) to a file automatically. 
Step 6 > Crate the exception.py file all exception handelling
Step 6 > Create the application.py file to call these functions of logging and execption and eventually other functoins in the pipeline too
Step 7 > In the components folder that was created in the template file, code the data ingestion pipeline in the data_ingestion.py file. This is the data ingestion component which will read the data from kaggle and split it into train and test datasets, that will be stored in the newly created artifacts folder as raw, train, and text files.
Step 8 > In the same way, create the data transformation .py file for EDA and feature scaling + encoding
Step 9 > To run step 8, we need changes to be made in Utils.py and application.py
Step 10 >  In the same way, create the model training.py file for training the train data set on various models and give the R2 score for each. We also need to make changes in the Utils.py and application.py
Step 11 > Goto dagshub and connect github repository, then connect with MLflow and add code to model training.py file. 
Step 12 >  add commands to set up your env with credentials form dagshub.
Step 13 > model results are logged here -https://dagshub.com/Hardika-Jain/ML_Project_E2E/experiments





