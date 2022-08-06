# Databricks notebook source
# MAGIC %md The purpose of this notebook is to download and set up the data we will use for the solution accelerator. Before running this notebook, make sure you have entered your own credentials for Kaggle and accepted the rules of this contest [dataset](https://www.kaggle.com/competitions/instacart-market-basket-analysis/rules).

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

# MAGIC %md 
# MAGIC Set Kaggle credential configuration values in the block below: You can set up a [secret scope](https://docs.databricks.com/security/secrets/secret-scopes.html) to manage credentials used in notebooks. For the block below, we have manually set up the `solution-accelerator-cicd` secret scope and saved our credentials there for internal testing purposes.

# COMMAND ----------

import os
# os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_username'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_username")

# os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_key'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_key")

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /databricks/driver
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle competitions download -c instacart-market-basket-analysis
# MAGIC unzip instacart-market-basket-analysis.zip
# MAGIC unzip aisles.csv.zip          
# MAGIC unzip departments.csv.zip     
# MAGIC unzip order_products__prior.csv.zip  
# MAGIC unzip order_products__train.csv.zip  
# MAGIC unzip orders.csv.zip          
# MAGIC unzip products.csv.zip        
# MAGIC unzip sample_submission.csv.zip 

# COMMAND ----------

# MAGIC %md Move the downloaded data to the folder used throughout the accelerator:

# COMMAND ----------

dbutils.fs.rm("dbfs:/tmp/instacart_wide_deep/", True)
dbutils.fs.mv("file:/databricks/driver/aisles.csv", "dbfs:/tmp/instacart_wide_deep/bronze/aisles/aisles.csv")
dbutils.fs.mv("file:/databricks/driver/departments.csv", "dbfs:/tmp/instacart_wide_deep/bronze/departments/departments.csv")
dbutils.fs.mv("file:/databricks/driver/order_products__prior.csv", "dbfs:/tmp/instacart_wide_deep/bronze/order_products/order_products__prior.csv")
dbutils.fs.mv("file:/databricks/driver/order_products__train.csv", "dbfs:/tmp/instacart_wide_deep/bronze/order_products/order_products__train.csv")
dbutils.fs.mv("file:/databricks/driver/orders.csv", "dbfs:/tmp/instacart_wide_deep/bronze/orders/orders.csv")
dbutils.fs.mv("file:/databricks/driver/products.csv", "dbfs:/tmp/instacart_wide_deep/bronze/products/products.csv")
