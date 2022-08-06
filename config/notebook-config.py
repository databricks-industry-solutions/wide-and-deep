# Databricks notebook source
# MAGIC %pip install tensorflow==2.9.1 # to avoid protobuf incompatibility with model serving

# COMMAND ----------

# We use 'instacart' as the database name to stay simple
# please use a personalized database name here if you wish to avoid interfering with other users who might be running this accelerator in the same workspace
database = 'instacart_wide_deep' 
spark.sql(f'CREATE DATABASE IF NOT EXISTS {database}')
spark.sql(f'USE {database}')
