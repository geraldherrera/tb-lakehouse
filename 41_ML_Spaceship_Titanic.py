# Databricks notebook source
# MAGIC %md
# MAGIC # Spaceship Titanic - Kaggle competition
# MAGIC
# MAGIC https://www.kaggle.com/competitions/spaceship-titanic
# MAGIC
# MAGIC This notebook serves as an introduction for the class about supervised learning.
# MAGIC Students are introduced to predictive modeling. They build a Scikit-Learn pipeline and fit a simple DecisionTreeClassifier on the Spaceship Titanic dataset. The model is evaluated with a kaggle submission.

# COMMAND ----------

import mlflow
import mlflow.sklearn

# Select the experiment to log the model
mlflow.set_experiment("/Users/gerald.herrera@he-arc.ch/ml_sandbox/titanic_demo")

# Enable autologging for scikit-learn models
mlflow.sklearn.autolog(
    log_models=True,
    registered_model_name="Titanic_DT"
)


# COMMAND ----------

# MAGIC %md
# MAGIC We load the train data. The PassengerId column is used as the index of the dataframe

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG dbw_dataplatform_dev_ghe_4363803328954077;
# MAGIC CREATE SCHEMA IF NOT EXISTS ml_sandbox;
# MAGIC USE SCHEMA  ml_sandbox;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE titanic_train
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT *
# MAGIC FROM read_files(
# MAGIC        '/Volumes/dbw_dataplatform_dev_ghe_4363803328954077/ml_sandbox/ml_volume/train.csv',
# MAGIC        format  => 'csv',
# MAGIC        header  => true
# MAGIC      );
# MAGIC
# MAGIC CREATE OR REPLACE TABLE titanic_test
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT *
# MAGIC FROM read_files(
# MAGIC        '/Volumes/dbw_dataplatform_dev_ghe_4363803328954077/ml_sandbox/ml_volume/test.csv',
# MAGIC        format  => 'csv',
# MAGIC        header  => true
# MAGIC      );
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np

train_spark = spark.table("dbw_dataplatform_dev_ghe_4363803328954077.ml_sandbox.titanic_train")

train = train_spark.toPandas()
train.set_index("PassengerId", inplace=True)

train

# COMMAND ----------

train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing Pipeline
# MAGIC
# MAGIC We identified null values in all columns. We will clean these by type.

# COMMAND ----------

train.isna().sum()

# COMMAND ----------

# pip install scikit-learn

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

train_spark = spark.table("dbw_dataplatform_dev_ghe_4363803328954077.ml_sandbox.titanic_train")

train = train_spark.toPandas()
train.set_index("PassengerId", inplace=True)

# Step 1: Define transformers for different column types
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean"))]
)

categorical_cols = ['HomePlanet', 'Destination', 'VIP', 'CryoSleep']
categorical_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder())
])

# Step 2: Create a ColumnTransformer that applies the transformations to the columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop' 
)

# Step 3: Assemble the preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# Fit and transform the DataFrame
X_train = preprocessing_pipeline.fit_transform(train)

preprocessing_pipeline

# COMMAND ----------

# Converting back to Pandas DataFrame
onehot_encoder_feature_names = list(preprocessing_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out())
column_order =  numerical_cols + onehot_encoder_feature_names

# Show the cleaned DataFrame
pd.DataFrame(X_train, columns=column_order, index=train.index)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree Classifier 
# MAGIC
# MAGIC We extend the pipeline with a decision tree classifier to predict the Transported variable.

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow

X = train.drop('Transported', axis=1)
y = train['Transported']

# Define the hyperparameters for the DecisionTreeClassifier
hyperparams = {
    'criterion': 'entropy',     # Function to measure the quality of a split
    'max_depth': 3,             # Limits the depth of the tree to prevent overfitting
    'min_samples_split': 20,    # The minimum number of samples required to split an internal node
    'min_samples_leaf': 10,     # The minimum number of samples required to be at a leaf node
    'random_state': 42          # Ensures reproducibility of the results
}

# Update the model pipeline with the new DecisionTreeClassifier parameters
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(**hyperparams))
])

with mlflow.start_run(run_name="dt_baseline"):
    # Fit the model
    model_pipeline.fit(X, y)
    # Log the hyperparameters
    train_acc = accuracy_score(y, model_pipeline.predict(X))
    mlflow.log_metric("train_accuracy", train_acc)

model_pipeline

# COMMAND ----------

# pip install matplotlib

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tree visualization
# MAGIC
# MAGIC We use matplotlib library to plot the tree we just fitted

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Extract the decision tree model
decision_tree_model = model_pipeline.named_steps['classifier']

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(decision_tree_model, 
          filled=True, 
          rounded=True,
          class_names=['Not Transported', 'Transported'],
          feature_names=column_order)  # Ensure 'column_order' matches the order of features in the trained model
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluation on Kaggle

# COMMAND ----------

#test_spark = spark.table("dbw_dataplatform_dev_ghe_4363803328954077.ml_sandbox.titanic_test")
#test = test_spark.toPandas()
#test.set_index("PassengerId", inplace=True)

#test

# COMMAND ----------

#X_test = test

#y_pred = model_pipeline.predict(X_test)

#kaggle_submission = pd.DataFrame(y_pred, columns=['Transported'], index=X_test.index)
#kaggle_submission

# COMMAND ----------

# Writing the submission DataFrame to a CSV file
#kaggle_submission.to_csv("Data/simple_decision_tree.csv", index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Call the model from the registry

# COMMAND ----------

import mlflow.sklearn

# 1. load the model from the registry
model_uri = "models:/Titanic_DT/@champion"
model = mlflow.sklearn.load_model(model_uri)

# 2. read the test data
test = (
    spark.table("dbw_dataplatform_dev_ghe_4363803328954077.ml_sandbox.titanic_test")
         .toPandas()
         .set_index("PassengerId")
)

# 3. select the feature columns
feature_cols = [c for c in test.columns
                if c != "Transported"]

X_test = test[feature_cols]

# 4. make predictions
y_pred = model.predict(X_test)

# 5. create a DataFrame for the results
import pandas as pd
result = pd.DataFrame({
    "PassengerId": X_test.index,
    "predicted_Transported": y_pred
}).set_index("PassengerId")

print(result.head(10))