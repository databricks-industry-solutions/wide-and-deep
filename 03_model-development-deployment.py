# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/wide-and-deep. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to train, evaluate & deploy a wide & deep collaborative filter recommender using features engineered in the prior notebook.  

# COMMAND ----------

# MAGIC %md ## Step 1: Prepare the Data
# MAGIC 
# MAGIC In our last notebook, both user and product features were prepared along with labels indicating whether a specific user-product combination was purchased in our training period.  Here, we will retrieve those data, combining them for input into our model:

# COMMAND ----------

# DBTITLE 1,Set database we use throughout this notebook
# MAGIC %run ./config/notebook-config

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as f
from pyspark.sql.types import *

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

import mlflow
from mlflow.tracking import MlflowClient

import platform

import numpy as np
import pandas as pd

import datetime
import os
import requests

# COMMAND ----------

# DBTITLE 1,Retrieve Features & Labels
# retrieve features and labels
product_features = spark.table('product_features')
user_features = spark.table('user_features')
labels = spark.table('labels')

# assemble full feature set
labeled_features = (
  labels
  .join(product_features, on='product_id')
  .join(user_features, on='user_id')
  )

# display results
display(labeled_features)

# COMMAND ----------

# MAGIC %md Because of the large number of features, we'll need to capture some metadata on our fields.  This metadata will help us setup our data inputs in later steps:

# COMMAND ----------

# DBTITLE 1,Capture Label & Feature Info
# identify label column
label_col = 'label'

# identify categorical feature columns
cat_features = ['aisle_id','department_id','user_id','product_id']

# capture keys for each of the categorical feature columns
cat_keys={}
for col in cat_features:
  cat_keys[col] = (
    labeled_features
      .selectExpr('{0} as key'.format(col))
      .distinct()
      .orderBy('key')
      .groupBy()
        .agg(f.collect_list('key').alias('keys'))
      .collect()[0]['keys']
    )

# all other columns (except id) are continuous features
num_features = labeled_features.drop(*(['id',label_col]+cat_features)).columns

# COMMAND ----------

# MAGIC %md Now we can split our data into training, validation & testing sets.  We pre-split this here versus dynamically splitting so that we might perform a stratified sample on the label.  The stratified sample will help ensure the under-represented positive class (indicating a specific product was purchased in the training period) is consistently present in our data splits:

# COMMAND ----------

# DBTITLE 1,Evaluate Positive Class Representation
instance_count = labeled_features.count()
positive_count = labels.filter(f.expr('label=1')).count()

print('{0:.2f}% positive class across {1} instances'.format(100 * positive_count/instance_count, instance_count))

# COMMAND ----------

# DBTITLE 1,Split Data into Training, Validation & Testing
# fraction to hold for training
train_fraction = 0.6

# sample data, stratifying on labels, for training
train = (
  labeled_features
    .sampleBy(label_col, fractions={0: train_fraction, 1: train_fraction})
  )

# split remaining data into validation & testing datasets (with same stratification)
valid = (
  labeled_features
    .join(train, on='id', how='leftanti') # not in()
    .sampleBy(label_col, fractions={0:0.5, 1:0.5})
  )

test = (
  labeled_features
    .join(train, on='id', how='leftanti') # not in()
    .join(valid, on='id', how='leftanti') # not in()
  )

# COMMAND ----------

# MAGIC %md The training, validation & testing datasets currently exist as Spark Dataframes and may be quite large.  Converting our data to a pandas Dataframe may result in an out of memory error, so instead, we'll convert our Spark Dataframe into a [Petastorm](https://petastorm.readthedocs.io/en/latest/) dataset. Petastorm is a library that caches Spark data to Parquet and provides high-speed, batched access to that data to libraries such as Tensorflow and PyTorch:
# MAGIC 
# MAGIC **NOTE** Petastorm may complain that a given cached file is too small.  Use the repartition() method to adjust the number of cached files generated with each dataset, but play with the count to determine the best number of files for your scenario.

# COMMAND ----------

# DBTITLE 1,Cache Data for Faster Access
# configure temp cache for petastorm files
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'file:///dbfs/tmp/instacart_wide_deep/pstorm_cache') # the file:// prefix is required by petastorm

# persist dataframe data to petastorm cache location
train_pstorm = make_spark_converter(train.repartition(4))  
valid_pstorm = make_spark_converter(valid.repartition(4)) 
test_pstorm = make_spark_converter(test.repartition(4)) 

# COMMAND ----------

# MAGIC %md To make the data in the Petastorm cache accessible, we will need to define specs that read the data and transform it into the format expected by Tensorflow.  This format requires features to be presented as a dictionary and the label to be presented as a scalar value:

# COMMAND ----------

# DBTITLE 1,Define Data Specs
def get_data_specs(epochs=1, batch_size=128):
  epochs = int(epochs)
  batch_size = int(batch_size)
  # define functions to transform data into req'ed format
  def get_input_fn(dataset_context_manager):
    
    # re-structure a row as ({features}, label)
    def _to_tuple(row): 
      features = {}
      for col in cat_features + num_features:
        features[col] = getattr(row, col)
      return features, getattr(row, label_col)
    
    def fn(): # called by estimator to perform row structure conversion
      return dataset_context_manager.__enter__().map(_to_tuple)
    
    return fn

  # access petastorm cache as tensorflow dataset
  train_ds = train_pstorm.make_tf_dataset(batch_size=batch_size)
  valid_ds = valid_pstorm.make_tf_dataset()
  
  # define spec to return transformed data for model training & evaluation
  train_spec = tf.estimator.TrainSpec(
                input_fn=get_input_fn(train_ds), 
                max_steps=int( (train_pstorm.dataset_size * epochs) / batch_size )
                )
  eval_spec = tf.estimator.EvalSpec(
                input_fn=get_input_fn(valid_ds)
                )
  
  return train_spec, eval_spec

# COMMAND ----------

# MAGIC %md We can verify our specs by retrieving a row as follows.  Note that the default batch size for the training (first) spec is 128 records:

# COMMAND ----------

# DBTITLE 1,Verify Spec
# retrieve specs
specs = get_data_specs()

# retrieve first batch from first (training) spec
next(
  iter(
    specs[0].input_fn().take(1)
    )
  )

# COMMAND ----------

# MAGIC %md ## Step 2: Define the Model
# MAGIC 
# MAGIC With our data in place, we can now define the wide & deep model.  For this, we will make use of [Tensorflow's DNNLinearCombinedClassifier estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier) which simplifies the definition of these kinds of models.
# MAGIC 
# MAGIC The feature inputs for the DNNLinearCombinedClassifier estimator are divided into those associated with a *wide*, linear model and those associated with a *deep* neural network.  The inputs to the wide model are the user and product ID combinations.  In this way, the linear model is being trained to memorize which products are purchased by which users.  These features may be brought into the model as simple categorical features [identified through an ordinal value](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_identity) or [hashed into a smaller number of buckets](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket).  The inclusion of a user-product [crossed hash](https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column) allows the model to better understand user-product combinations:
# MAGIC 
# MAGIC **NOTE** Much of the logic that follows is encapsulated in functions.  This will make distributed processing occurring later in the notebook easier to implement and is fairly standard for most Tensorflow implementations.

# COMMAND ----------

# DBTITLE 1,Define Wide Features
def get_wide_features():

  wide_columns = []

  # user_id
  #wide_columns += [tf.feature_column.categorical_column_with_identity(
  #    key='user_id', 
  #    num_buckets=np.max(cat_keys['user_id'])+1 # create one bucket for each value from 0 to max
  #    )]
  wide_columns += [
    tf.feature_column.categorical_column_with_hash_bucket(
       key='user_id', 
       hash_bucket_size=1000,
       dtype=tf.dtypes.int64# create one bucket for each value from 0 to max
       )]

  # product_id
  #wide_columns += [
  #  tf.feature_column.categorical_column_with_identity(
  #    key='product_id', 
  #    num_buckets=np.max(cat_keys['product_id'])+1 # create one bucket for each value from 0 to max
  #    )]
  wide_columns += [
    tf.feature_column.categorical_column_with_hash_bucket(
       key='product_id', 
       hash_bucket_size=100,
       dtype=tf.dtypes.int64 # create one bucket for each value from 0 to max
       )]

  # user-product cross-column (set column spec to ensure presented as int64)
  wide_columns += [
    tf.feature_column.crossed_column(
      [ tf.feature_column.categorical_column_with_identity(key='user_id', num_buckets=np.max(cat_keys['user_id'])+1),
        tf.feature_column.categorical_column_with_identity(key='product_id', num_buckets=np.max(cat_keys['product_id'])+1)
        ], 
      hash_bucket_size=1000
      )] 

  return wide_columns

# COMMAND ----------

# MAGIC %md The feature inputs for the deep (neural network) component of the model are the features that describe our users and products in more generalized ways. By avoiding specific user and product IDs, the deep model is trained to learn attributes that signal preferences between users and products. For the categorical features, an [embedding](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column) is used to succinctly capture the feature data.  The number of dimensions in the embedding is based on guidance in [this tutorial](https://tensorflow2.readthedocs.io/en/stable/tensorflow/g3doc/tutorials/wide_and_deep/):

# COMMAND ----------

# DBTITLE 1,Define Deep Features
def get_deep_features():
  
  deep_columns = []

  # categorical features
  for col in cat_features:

    # don't use user ID or product ID
    if col not in ['user_id','product_id']:

      # base column definition
      col_def = tf.feature_column.categorical_column_with_identity(
        key=col, 
        num_buckets=np.max(cat_keys[col])+1 # create one bucket for each value from 0 to max
        )

      # define embedding on base column def
      deep_columns += [tf.feature_column.embedding_column(
                          col_def, 
                          dimension=int(np.max(cat_keys[col])**0.25)
                          )] 

  # continuous features
  for col in num_features:
    deep_columns += [tf.feature_column.numeric_column(col)]  
    
  return deep_columns

# COMMAND ----------

# MAGIC %md With our features defined, we can now assemble the estimator:
# MAGIC 
# MAGIC **NOTE** The optimizers are passed as classes to address an issue identified [here](https://stackoverflow.com/questions/58108945/cannot-do-incremental-training-with-dnnregressor).

# COMMAND ----------

# DBTITLE 1,Assemble Model
def get_model(hidden_layers, hidden_layer_nodes_initial_count, hidden_layer_nodes_count_decline_rate, dropout_rate):  
  
  # determine hidden_units structure
  hidden_units = [None] * int(hidden_layers)
  for i in range(int(hidden_layers)):
    # decrement the nodes by the decline rate
    hidden_units[i] = int(hidden_layer_nodes_initial_count * (hidden_layer_nodes_count_decline_rate**i))
 
  # get features
  wide_features = get_wide_features()
  deep_features = get_deep_features()
    
  # define model
  estimator = tf.estimator.DNNLinearCombinedClassifier(
    linear_feature_columns=wide_features,
    linear_optimizer=tf.keras.optimizers.Ftrl,
    dnn_feature_columns=deep_features,
    dnn_hidden_units=hidden_units,
    dnn_dropout=dropout_rate,
    dnn_optimizer=tf.keras.optimizers.Adagrad
    )

  return estimator

# COMMAND ----------

# MAGIC %md ## Step 3: Tune the Model
# MAGIC 
# MAGIC To tune the model, we need to define an evaluation metric.  By default, the DNNLinearCombinedClassifier seeks to minimize the [softmax (categorical) cross entropy](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits) metric which examines the distance between a predicted class probability and the actual class label.  (You can think of this metric as seeking more accurate and confident class predictions.)
# MAGIC 
# MAGIC We'll tune our model around this metric but it might be nice to provide a more traditional metric to assist us with evaluation of the end result.  For recommenders where the goal is to present products in order from most likely to least likely to be selected, [mean average precision @ k (MAP@K)](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf) is often used.  This metric examines the average precision associated with a top-*k* number of recommendations.  The closer the value of MAP@K to 1.0, the better aligned those recommendations are with a customer's product selections. 
# MAGIC 
# MAGIC To calculate MAP@K, we are [repurposing code presented by NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Recommendation/WideAndDeep/utils/metrics.py) with their implementation of a wide and deep recommender for ad-placement:

# COMMAND ----------

# DBTITLE 1,Define MAP@K Metric
# Adapted from: https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Recommendation/WideAndDeep/utils/metrics.py
def map_custom_metric(features, labels, predictions):
  
  user_ids = tf.reshape(features['user_id'], [-1])
  predictions = predictions['probabilities'][:, 1]

  # sort user IDs 
  sorted_ids = tf.argsort(user_ids)
  
  # resort values to align with sorted user IDs
  user_ids = tf.gather(user_ids, indices=sorted_ids)
  predictions = tf.gather(predictions, indices=sorted_ids)
  labels = tf.gather(labels, indices=sorted_ids)

  # get unique user IDs in dataset
  _, user_ids_idx, user_ids_items_count = tf.unique_with_counts(
      user_ids, 
      out_idx=tf.int64
      )
  
  # remove any user duplicates
  pad_length = 30 - tf.reduce_max(user_ids_items_count)
  pad_fn = lambda x: tf.pad(x, [(0, 0), (0, pad_length)])
  preds = tf.RaggedTensor.from_value_rowids(
      predictions, user_ids_idx).to_tensor()
  labels = tf.RaggedTensor.from_value_rowids(
      labels, user_ids_idx).to_tensor()
  labels = tf.argmax(labels, axis=1)

  # calculate average precision at k
  return {
      'map@k': tf.compat.v1.metrics.average_precision_at_k(
          predictions=pad_fn(preds),
          labels=labels,
          k=10,
          name="streaming_map")
        }

# COMMAND ----------

# MAGIC %md We can now bring together all of our logic to define our model:

# COMMAND ----------

# DBTITLE 1,Define Training & Evaluation Logic
def train_and_evaluate_model(hparams):
  
  # retrieve the basic model
  model = get_model(
    hparams['hidden_layers'], 
    hparams['hidden_layer_nodes_initial_count'], 
    hparams['hidden_layer_nodes_count_decline_rate'], 
    hparams['dropout_rate']
    )
  
  # add map@k metric
  model = tf.estimator.add_metrics(model, map_custom_metric)
  
  # retrieve data specs
  train_spec, eval_spec = get_data_specs( int(hparams['epochs']), int(hparams['batch_size']))
  
  # train and evaluate
  results = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
  
  # return loss metric
  return {'loss': results[0]['loss'], 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md We will log runs to an mlflow experiment in the user's own folder:

# COMMAND ----------

useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/wide_and_deep"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# MAGIC %md We can now give our model a test run, just to make sure all the moving parts are working together:

# COMMAND ----------

# DBTITLE 1,Perform Test Run
hparams = {
  'hidden_layers':2,
  'hidden_layer_nodes_initial_count':100,
  'hidden_layer_nodes_count_decline_rate':0.5,
  'dropout_rate':0.25,
  'epochs':1,
  'batch_size':128
  }

train_and_evaluate_model(hparams)

# COMMAND ----------

# MAGIC %md With a successful test run completed, let's now perform hyperparameter tuning on the model. We will use [hyperopt](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt) to ensure this work is distributed in a manner that allows us to manage the total time required for this operation.
# MAGIC 
# MAGIC Regarding the hyperparameters, we will play with the number of hidden units as well as the drop-out rate for the deep neural-network portion of the model.  While we will have the option to tune the number of epochs and the batch size for training, we will leave those set to fixed values at this time:

# COMMAND ----------

# DBTITLE 1,Define Hyperparameter Search Space
search_space = {
  'hidden_layers': hp.quniform('hidden_layers', 1, 5, 1)  # determines number of hidden layers
  ,'hidden_layer_nodes_initial_count': hp.quniform('hidden_layer_nodes_initial', 50, 201, 10)  # determines number of nodes in first hidden layer
  ,'hidden_layer_nodes_count_decline_rate': hp.quniform('hidden_layer_nodes_count_decline_rate', 0.0, 0.51, 0.05) # determines how number of nodes decline in layers below first hidden layer
  ,'dropout_rate': hp.quniform('dropout_rate', 0.0, 0.51, 0.05)
  ,'epochs': hp.quniform('epochs', 3, 4, 1) # fixed value for now
  ,'batch_size': hp.quniform('batch_size', 128, 129, 1) # fixed value for now
  }

# COMMAND ----------

# DBTITLE 1,Perform Hyperparameter Search
argmin = fmin(
  fn=train_and_evaluate_model,
  space=search_space,
  algo=tpe.suggest,
  max_evals=100,
  trials=SparkTrials(parallelism=sc.defaultParallelism) # set to the number of executors for CPU-based clusters OR number of workers for GPU-based clusters
  )

# COMMAND ----------

# DBTITLE 1,Show Optimized Hyperparameters
space_eval(search_space, argmin)

# COMMAND ----------

# MAGIC %md ## Step 4: Evaluate the Model
# MAGIC 
# MAGIC Based on our optimized parameters, we can now train a final version of our model and explore the metrics associated with it:

# COMMAND ----------

# DBTITLE 1,Train Optimized Model
model = get_model(
    hparams['hidden_layers'], 
    hparams['hidden_layer_nodes_initial_count'], 
    hparams['hidden_layer_nodes_count_decline_rate'], 
    hparams['dropout_rate']
    )
model = tf.estimator.add_metrics(model, map_custom_metric)

train_spec, eval_spec = get_data_specs(hparams['epochs'], hparams['batch_size']) 

results = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

# COMMAND ----------

# DBTITLE 1,Review Validation Results
results[0]

# COMMAND ----------

# MAGIC %md Using our test data, which the model did not see during hyperparameter tuning, we can better assess model performance.  Our test data, also stored in Petastorm, requires access to a function to re-organize it for evaluation.  In addition, we need to explicitly define the number of data steps over which the data should be evaluated (or the evaluation step will run indefinitely): 

# COMMAND ----------

# DBTITLE 1,Evaluate Using Test Data
# Borrowed from get_data_specs() (defined above)
# ---------------------------------------------------------
# define functions to transform data into req'ed format
def get_input_fn(dataset_context_manager):

  def _to_tuple(row): # re-structure a row as ({features}, label)
    features = {}
    for col in cat_features + num_features:
      features[col] = getattr(row, col)
    return features, getattr(row, label_col)

  def fn(): # called by estimator to perform row structure conversion
    return dataset_context_manager.__enter__().map(_to_tuple)

  return fn
# ---------------------------------------------------------

# define batch size and number of steps
batch_size = 128
steps = int(test_pstorm.dataset_size/batch_size)

# retrieve test data
test_ds = test_pstorm.make_tf_dataset(batch_size=batch_size)

# evaulate against test data
results = model.evaluate(get_input_fn(test_ds), steps=steps)

# COMMAND ----------

# DBTITLE 1,Review Test Results
# show results
results

# COMMAND ----------

# MAGIC %md Our model appears to produce similar results for the testing holdout.  We should feel confident moving it into the application infrastructure for live testing.

# COMMAND ----------

# MAGIC %md ## Step 5: Deploy the Model
# MAGIC 
# MAGIC With our model trained and evaluated, we now need to move it into our application infrastructure.  To do this, we will need to persist the model in a manner that enables deployment. For this, we'll make use of MLflow, but before we can do that, we'll need to persist the model using Tensorflow's built-in functionality. This will make it easier for MLflow to pickup the pickled model later on:

# COMMAND ----------

# DBTITLE 1,Temporarily Export Tensorflow Model
# get features
wide_features = get_wide_features()
deep_features = get_deep_features()

# use features to generate an input specification
feature_spec = tf.feature_column.make_parse_example_spec(
    wide_features + deep_features
    )

# make function to apply specification to incoming data
fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    feature_spec
    )

# export the model
saved_model_path = model.export_saved_model(
    export_dir_base='/dbfs/tmp/exported_model',
    serving_input_receiver_fn=fn
    ).decode("utf-8")

# COMMAND ----------

# MAGIC %md The Tensorflow model expects data in a particular format and will return a number of values with each prediction.  We'll provide MLflow a custom wrapper for the which will pickup the pickled Tensorflow model, convert incoming data into a format understood by that model and return positive class probabilities with each prediction:

# COMMAND ----------

# DBTITLE 1,Define Wrapper for Model
# custom wrapper to align model with mlflow
class Recommender(mlflow.pyfunc.PythonModel):
  
  # The code snippet in this cell can be reused on new datasets with some modifications to match the feature types of your dataset. 
  # see docs: https://www.tensorflow.org/tutorials/load_data/tfrecord
  def _convert_inputs(self, inputs_pd):

    proto_tensors = []

    # for each row in the pandas dataframe
    for i in range(len(inputs_pd)):

      # translate field values into features
      feature = dict()
      for field in inputs_pd:
        if field not in ['id','aisle_id','department_id','user_id','product_id']:
          feature[field] = tf.train.Feature(float_list=tf.train.FloatList(value=[inputs_pd[field][i]]))
        else: 
          feature[field] = tf.train.Feature(int64_list=tf.train.Int64List(value=[inputs_pd[field][i]]))

      # convert rows into expected format
      proto = tf.train.Example(features=tf.train.Features(feature=feature))
      proto_string = proto.SerializeToString()
      proto_tensors.append(tf.constant([proto_string]))

    return proto_tensors
  
  # load saved model upon initialization
  def load_context(self, context):
    self.model = mlflow.tensorflow.load_model(context.artifacts['model'])  # retrieve the unaltered tensorflow model (persisted as an artifact)

  # logic to return scores
  def predict(self, context, model_input):
    # convert inputs into required format
    proto = self._convert_inputs(model_input)
    # score inputs
    results_list = []
    for p in proto:
      results_list.append(self.model(p))
    # retrieve positive class probability as score 
    ret = [item['probabilities'][0, 1].numpy() for item in results_list]
    return ret

# COMMAND ----------

# MAGIC %md Our model will expect a large number of features to be passed to it.  To help clarify the data structure requirements, we'll create a [sample dataset to persist with the model](https://www.mlflow.org/docs/latest/models.html#input-example):

# COMMAND ----------

# DBTITLE 1,Construct Sample Input Dataset
# use user 123 as sample user
user_id = spark.createDataFrame([(123,)], ['user_id'])

# get features for user and small number of products
sample_pd = (
  user_id
    .join(user_features, on='user_id') # get user features
    .crossJoin(product_features.limit(5)) # get product features (for 5 products)
  ).toPandas()

# show sample
sample_pd

# COMMAND ----------

# MAGIC %md Now we can finally persist our model to MLflow. We'll register the model using a recognizable name (to help us with a later step) and include information about the library dependencies for this model:

# COMMAND ----------

# DBTITLE 1,Identify Model Name
model_name='recommender'

# COMMAND ----------

# DBTITLE 1,Persist Models to mlflow
# libraries for these models
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      f'python={platform.python_version()}',
      'pip',
      {
        'pip': [
          'mlflow',
          #f'tensorflow-gpu=={tf.__version__}',  # gpu-version
          f'tensorflow-cpu=={tf.__version__}',   # cpu-version
          'tensorflow-estimator',
        ],
      },
    ],
    'name': 'recommender_env'
}

# create an experiment run under which to store these models
with mlflow.start_run(run_name=model_name) as run:
  
  # log the tensorflow model to mlflow
  tf_meta_graph_tags = [tag_constants.SERVING]
  tf_signature_def_key='predict'
  
  # persist the original tensorflow model
  mlflow.tensorflow.log_model(
    tf_saved_model_dir=saved_model_path, 
    tf_meta_graph_tags=tf_meta_graph_tags, 
    tf_signature_def_key=tf_signature_def_key,
    artifact_path='model',
    conda_env=conda_env
    )
  
  # retrieve artifact path for the tensorflow model just logged
  artifacts = {
    # Replace the value with the actual model you logged
    'model': mlflow.get_artifact_uri() + '/model'
  }

  # record the model with the custom wrapper with the tensorflow model as its artifact
  mlflow_pyfunc_model_path = 'recommender_mlflow_pyfunc'
  mlflow.pyfunc.log_model(
    artifact_path=mlflow_pyfunc_model_path, 
    python_model=Recommender(), 
    artifacts=artifacts,
    conda_env=conda_env, 
    input_example=sample_pd)

  # register last deployed model with mlflow model registry
  mv = mlflow.register_model(
      'runs:/{0}/{1}'.format(run.info.run_id, mlflow_pyfunc_model_path),
      model_name
      )
  
  # record model version for next step
  model_version = mv.version

# COMMAND ----------

# MAGIC %md Towards the end of the steps in the last cell, we register the model with the MLflow registry.  The [MLflow registry](https://www.mlflow.org/docs/latest/model-registry.html) provides support for the promotion of models from an initial state to staging to production and then to archival. Organizations may build a workflow to enable models to be evaluated before being designated as the current production instance. For our purposes, we'll push our just published model directly to production without testing and coordination with dependent systems, understanding this is not recommended practice outside of a demonstration:

# COMMAND ----------

# DBTITLE 1,Promote Model to Production Status
client = mlflow.tracking.MlflowClient()

# archive any production versions of the model from prior runs
for mv in client.search_model_versions("name='{0}'".format(model_name)):
  
    # if model with this name is marked production
    if mv.current_stage.lower() == 'production':
      # mark is as archived
      client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage='archived'
        )
      
# transition newly deployed model to production stage
client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='Production'
  )     

# COMMAND ----------

# MAGIC %md We now have a model in the MLflow registry that can be employed to score user-product combinations.  We can envision a scenario where an application page might present a subset of products associated with a search term or category selection, send features for those products along with user features to the model, and receive scores which could then be used to order the products on the page. 
# MAGIC 
# MAGIC In this scenario, the serving of the model may be done through a microservice hosted in [Azure ML](https://www.mlflow.org/docs/latest/python_api/mlflow.azureml.html), [AWS Sagemaker](https://www.mlflow.org/docs/latest/python_api/mlflow.sagemaker.html) or even Databricks itself through its [Model Serving](https://docs.databricks.com/applications/mlflow/model-serving.html) capabilities. Many other variations on deployment may be possible given that essentially MLflow is deploying the model to a [Docker image](https://www.mlflow.org/docs/latest/cli.html#mlflow-models-build-docker) and exposing it via a pre-defined REST endpoint.  Given this, technologies such as [Kubernetes](https://www.mlflow.org/docs/latest/projects.html#run-an-mlflow-project-on-kubernetes-experimental) come into the picture as viable deployment paths.
# MAGIC 
# MAGIC To get a sense of how this might work, we'll make use of Databricks Model Serving.  To access the Serving UI, switch into the Databricks UI's Machine Learning interface by clicking the drop-down in the upper left-hand corner of your screen. By clicking on the Models icon  <img src='https://brysmiwasb.blob.core.windows.net/demos/images/widedeep_models_icon.PNG' width=50>  in the left-hand side of the Databricks UI, we can locate the model registered in the previous steps.  Clicking on that model, we are presented two tabs: Details & Serving.  Clicking the Serving tab, we can select the <img src='https://brysmiwasb.blob.core.windows.net/demos/images/widedeep_enableserving_button.PNG'> button to launch a small, single-node cluster to host the model.
# MAGIC 
# MAGIC Once clicked, select the production version of the model you wish to call.  Note the Model URL presented with the selected model version.  You may then submit data to the REST API associated with this model using the code below.
# MAGIC 
# MAGIC Please note that the single-node cluster running this model must transition from the **Pending** to the **Running** state before requests will be accepted.  Also, this cluster will run indefinitely unless you actively return to the Serving tab and select **Stop** next to its status.
# MAGIC 
# MAGIC Finally, the REST API presented by Databricks Model Serving is secured using a [Databricks Personal Access Token](https://docs.databricks.com/dev-tools/api/latest/authentication.html). Here we retrieve a personal access token from the notebook environment.

# COMMAND ----------

# DBTITLE 1,Retrieve User-Product Combinations to Score
user_id = 123 # user 123
aisle_id = 32 # packaged produce department

# retrieve user features
user_features = (
  spark
    .table('user_features')
    .filter(f.expr('user_id={0}'.format(user_id)))
  )

# retrieve features for product in department
product_features = (
  spark
    .table('product_features')
    .filter('aisle_id={0}'.format(aisle_id))
  )

# combine for scoring
user_products = (
  user_features
    .crossJoin(product_features)
    .toPandas()
    )

# show sample of feature set
user_products.head(5)

# COMMAND ----------

# DBTITLE 1,Retrieve Scores
personal_access_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
databricks_instance = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None) 
model_name = 'recommender'

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(databricks_instance, personal_access_token, model_name, dataset):
  url = f'{databricks_instance}/model/{model_name}/Production/invocations'
  headers = {'Authorization': f'Bearer {personal_access_token}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# call REST API for scoring
# score_model(databricks_instance, personal_access_token, model_name, user_products) # scoring works once model serving is deployed

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
