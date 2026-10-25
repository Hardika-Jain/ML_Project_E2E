Skincare Product Recommendation Prediction: End-to-End ML Project

An end-to-end machine learning pipeline that predicts whether a customer with a given skin type and tone will recommend a skincare or makeup product, using only the product's brand and price plus the customer's skin type and skin tone.

It covers the full workflow: data ingestion, cleaning and feature engineering, training and tuning eight classifiers, and experiment tracking with MLflow on DagsHub.

<Problem Statement>

Given a product's brand and price, and a customer's skin type and skin tone, will that customer recommend the product (is_recommended = 1) or not (= 0)?

This is a binary classification problem. It could support things like "customers with oily skin tend to recommend Product X 85% of the time," or a recommender that flags which products are likely to satisfy a particular kind of customer before they buy.

This project predicts historical recommendation behavior. It does not determine medical or clinical suitability of any product.

<Dataset>

Sephora Products and Skincare Reviews from Kaggle, downloaded automatically with kagglehub when the pipeline runs.
(https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

The pipeline merges the review files (reviews_*.csv) with product_info.csv on product_id, then keeps only the columns it needs:

Column	           Role
is_recommended	   Target (0 / 1)
skin_type	       Feature (one-hot encoded)
skin_tone	       Feature (ordinal encoded, light to dark)
brand_name_product Feature (frequency encoded)
price_usd_product  Feature (log-transformed, then standardized)
product_id	       Used for grouping the split, not a model feature

<Pipeline>

Kaggle data
   ↓
Data Ingestion        merge reviews + products, grouped train/test split
   ↓
Data Transformation   clean, log + scale price, encode skin tone / skin type / brand
   ↓
Model Training        8 classifiers, GridSearchCV tuning for each
   ↓
Best model selection  highest test accuracy (must be ≥ 0.60)
   ↓
MLflow + DagsHub      params, metrics, and registered model
   ↓
Saved artifacts       artifacts/model.pkl, artifacts/preprocessor.pkl

Details worth knowing:

1. Leak-free split - GroupShuffleSplit (80/20, random_state=42) keeps all reviews of a product entirely in train or entirely in test.
2. No preprocessing leakage - The scaler and encoders are fit on the training set only, then applied to the test set. Brands unseen in training get frequency 0.
3. Tuning - Every model is tuned with GridSearchCV (3-fold CV), refit on the full training set, then scored on the held-out test set.
4. Quality gate - The run fails if the best model's accuracy is below 0.60.

<Models evaluated>

Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, K-Neighbors, XGBoost, CatBoost.


<Project Structure>

├── application.py                  # Entry point: runs the full pipeline
├── requirements.txt                # Python dependencies
├── setup.py                        # Packaging (reads requirements.txt)
├── template.py                     # One-off script that scaffolds the folder layout
├── notebook/
│   └── eda.ipynb                   # Exploratory data analysis
└── src/ML_Project_Hardika/
    ├── components/
    │   ├── data_ingestion.py       # Download, merge, split
    │   ├── data_transformation.py  # Clean, scale, encode, save preprocessor
    │   ├── model_training.py       # Train, tune, select, log to MLflow
    │   └── model_evaluation.py     
    ├── implementation/
    │   ├── training_pipeline.py    
    │   └── prediction_pipeline.py  
    ├── logger.py                   # Timestamped file logging
    ├── exception.py                # CustomException with file/line details
    └── utils.py                    # save_object, load_object, evaluate_model


<Results>

Accuracy of each model on the held-out test set:

Model	Accuracy
Logistic Regression	63.08%
Decision Tree	63.08%
Random Forest	63.08%
Gradient Boosting	63.08%
XGBoost	63.08%
CatBoost	63.08%
K-Neighbors	61.50%

Best model: Logistic Regression

Metric	    Value
Accuracy	63.08%
Precision	63.08%
Recall	    100.00%
F1 Score	77.36%
ROC-AUC	    55.24%

<Interpreting these results:>

1. Precision equal to accuracy, together with 100% recall, means the best model predicts "recommended" for every row. Its accuracy simply matches the share of positive reviews in the test set, so it is effectively a majority-class baseline rather than a learned signal.
2. The ROC-AUC of 55.24% (50% is random guessing) confirms weak discrimination.
3. Most models landing on the same accuracy points the same way: the four available features (brand, price, skin type, skin tone) carry little information about whether someone recommends a product.

Improving this likely needs richer features, such as review text, rating, product category or ingredients, and handling class imbalance (class weights, resampling, or threshold tuning).






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
Step 12 > add commands to set up your env with credentials form dagshub.
Step 13 > model results are logged here -https://dagshub.com/Hardika-Jain/ML_Project_E2E/experiments





