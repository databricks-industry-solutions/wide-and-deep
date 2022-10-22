# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/wide-and-deep. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to engineer the features we will use to build a wide and deep collaborative filter recommender.  

# COMMAND ----------

# MAGIC %md ## Step 1: Calculate Base Metrics
# MAGIC 
# MAGIC Model-based collaborative filters use user and product features to predict a future purchase or interaction.  The [wide-and-deep model](https://arxiv.org/abs/1606.07792) does this recognizing that a customer's future purchases are likely to be a result of prior user-product interactions as well as general patterns surrounding user-product preferences. In this regard, it balances a specific user's preference for particular products with more generalized preferences that would influence the purchase of new, *i.e.* previously unpurchased, items.  
# MAGIC 
# MAGIC For the wide-part of the model, the features are straightforward: we make use of the user and the product IDs to *memorize* preferences.  For the deep-part of the model, we need a variety of features that describe the user and the products to enable *generalization*.  These features are derived from metrics extracted from our historical data, labeled here as the *prior* evaluation set:

# COMMAND ----------

# DBTITLE 1,Set database we use throughout this notebook
# MAGIC %run ./config/notebook-config

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql import functions as f

# COMMAND ----------

# DBTITLE 1,Retrieve Prior Orders
order_details_ = spark.table('order_details').cache()
prior_order_details = order_details_.filter(f.expr("eval_set='prior'"))

# COMMAND ----------

# MAGIC %md Many of our deep-features will be calculated based on orders place a fixed number of days prior to the last order placed.  We will arbitrarily set these intervals as 30-days, 180-days and 360-days prior:

# COMMAND ----------

# DBTITLE 1,Set Days-Prior Boundaries
prior_days = [30, 180, 360]

# COMMAND ----------

# MAGIC %md We can now calculate counts for various distinct elements observed within these prior windows.  These global metrics will be used to convert totals derived below into ratios in later steps. Because of the redundant nature of the metric definitions, we will iteratively construct these metrics before asking Spark to resolve them for us:

# COMMAND ----------

# DBTITLE 1,Calculate Global Metrics
# calculate metrics for the following fields and time intervals
aggregations = []
for column in ['order_id', 'user_id', 'product_id', 'department_id', 'aisle_id']:
  for prior_day in prior_days:
    
    # count distinct instances in the field during this time-range
    aggregations += [
      f.countDistinct(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('global_cnt_distinct_{1}_last_{0}_days'.format(prior_day, column))]
    
# execute metric definitions
global_metrics = (
  prior_order_details
  ).agg(*aggregations)

# show results
display(global_metrics)

# COMMAND ----------

# MAGIC %md We now calculate product-specific metrics:

# COMMAND ----------

# DBTITLE 1,Calculate Product Metrics
# calculate metrics for the following fields and time intervals
aggregations = []

# distinct count metrics
for column in ['order_id', 'user_id']:
  for prior_day in prior_days:
    
    aggregations += [
      f.countDistinct(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('product_cnt_distinct_{1}_last_{0}_days'.format(prior_day, column))]

# occurrence count metrics
for column in ['reordered', 1]:
  for prior_day in prior_days:
    
    aggregations += [
      f.sum(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('product_sum_{1}_last_{0}_days'.format(prior_day, column))]
    
# get last assigned department & aisle for each product
  product_cat = (
    prior_order_details
      .select('product_id','aisle_id','department_id','order_id')
      .withColumn('aisle_id', f.expr('LAST(aisle_id) OVER(PARTITION BY product_id ORDER BY order_id)'))
      .withColumn('department_id', f.expr('LAST(department_id) OVER(PARTITION BY product_id ORDER BY order_id)'))
      .select('product_id','aisle_id','department_id')
      .distinct()
    )

# execute metric definitions
product_metrics = (
  prior_order_details
    .groupBy('product_id')
      .agg(*aggregations)
    .join(product_cat, on='product_id')
  )

# show results
display(product_metrics)

# COMMAND ----------

# MAGIC %md And now we calculate user-specific metrics:

# COMMAND ----------

# DBTITLE 1,Calculate User Metrics 
# calculate metrics for the following fields and time intervals
aggregations = []

# distinct count metrics
for column in ['order_id', 'product_id', 'department_id', 'aisle_id']:
  for prior_day in prior_days:
    
    aggregations += [
      f.countDistinct(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('user_cnt_distinct_{1}_last_{0}_days'.format(prior_day, column))]    

# occurrence count metrics
for column in ['reordered', 1]:
  for prior_day in prior_days:
    
    aggregations += [
      f.sum(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('user_sum_{1}_last_{0}_days'.format(prior_day, column))]
    
# execute metric definitions  
user_metrics = (
  prior_order_details
    .groupBy('user_id')
      .agg(*aggregations)
  )

# show results
display(user_metrics)

# COMMAND ----------

# MAGIC %md ## Step 2: Calculate Features
# MAGIC 
# MAGIC With our metrics calculated, we can now use these to generate product-specific features.  We will persist our product-specific features separately from user-features to enable easier data assembly later:

# COMMAND ----------

# DBTITLE 1,Product-Specific Features
# calculate product specific features
product_feature_definitions = []
for prior_day in prior_days:
  
  # distinct users associated with a product within some number of prior days
  product_feature_definitions += [f.expr('product_cnt_distinct_user_id_last_{0}_days/global_cnt_distinct_user_id_last_{0}_days as product_shr_distinct_users_last_{0}_days'.format(prior_day))]
  
  # distinct orders associated with a product within some number of prior days
  product_feature_definitions += [f.expr('product_cnt_distinct_order_id_last_{0}_days/global_cnt_distinct_order_id_last_{0}_days as product_shr_distinct_orders_last_{0}_days'.format(prior_day))]
  
  # product reorders within some number of prior days
  product_feature_definitions += [f.expr('product_sum_reordered_last_{0}_days/product_sum_1_last_{0}_days as product_shr_reordered_last_{0}_days'.format(prior_day))]
  
# execute features
product_features = (
  product_metrics
    .join(global_metrics) # cross join to a single row
    .select(
      'product_id',
      'aisle_id',
      'department_id',
      *product_feature_definitions
      )
  ).na.fill(0) # fill any missing values with 0s

# persist data
(
product_features
  .write
  .format('delta')
  .mode('overwrite')
  .option('overwriteSchema','true')
  .saveAsTable('product_features')
)

# show results
display(spark.table('product_features'))

# COMMAND ----------

# MAGIC %md Similarly, we can calculate user-specific features and persist these for later use:

# COMMAND ----------

# DBTITLE 1,User-Specific Features
# calculate user-specific order metrics
median_cols = ['lines_per_order', 'days_since_prior_order']
approx_median_stmt = [f.expr(f'percentile_approx({col}, 0.5)').alias(f'user_med_{col}') for col in median_cols]

user_order_features = (
  prior_order_details
    .groupBy('user_id','order_id')  # get order-specific details for each user
      .agg(
        f.first('days_since_prior_order').alias('days_since_prior_order'),
        f.count('*').alias('lines_per_order')        
        )
    .groupBy('user_id') # get median values across user orders
      .agg(*approx_median_stmt)
  ).na.fill(0)

# calculate user overall features
user_feature_definitions = []
user_drop_columns = []

for prior_day in prior_days:
  user_feature_definitions += [f.expr('user_sum_reordered_last_{0}_days/user_sum_1_last_{0}_days as user_shr_reordered_last_{0}_days'.format(prior_day))]
  user_drop_columns += ['user_sum_reordered_last_{0}_days'.format(prior_day)]
  user_drop_columns += ['user_sum_1_last_{0}_days'.format(prior_day)]
  
# assemble final set of user features
user_features = (
  user_metrics
    .join(user_order_features, on=['user_id'])
    .select(
      f.expr('*'),
      *user_feature_definitions
      )
    .drop(*user_drop_columns)
  ).na.fill(0)

# persist data
(
user_features
  .write
  .format('delta')
  .mode('overwrite')
  .option('overwriteSchema','true')
  .saveAsTable('user_features')
)

# show user features
display(spark.table('user_features'))

# COMMAND ----------

# MAGIC %md # Step 3: Generate Labels
# MAGIC 
# MAGIC Now we need to label each user-product pair observed across the dataset.  We will identify each user-product entry with a 1 if that record is something bought by the customer in his or her last purchase, *i.e.* during the *training* period, and a 0 if not:
# MAGIC 
# MAGIC **NOTE** We elected not to examine every user-product combination and instead limited our dataset to those combinations which occurred in the prior or training periods.  This is a choice that others may wish to revisit for their datasets.

# COMMAND ----------

# DBTITLE 1,Identify User-Product Combinations in Last Purchase
train_labels = (
  order_details_
    .filter(f.expr("eval_set='train'"))
    .select('user_id', 'product_id')
    .distinct()
    .withColumn('label', f.lit(1))
     )

labels = (
  prior_order_details
    .select('user_id','product_id')
    .distinct()
    .join(train_labels, on=['user_id','product_id'], how='fullouter') # preserve all user-product combinations observed in either period
    .withColumn('label',f.expr('coalesce(label,0)'))
    .select('user_id','product_id','label')
    .withColumn('id', f.monotonically_increasing_id())
  )
  
(
  labels
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('labels')
  )
  
display(spark.table('labels'))

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
