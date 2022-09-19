# Introduction 

Collaborative filters leverage similarities between users to make recommendations:

<img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_collabrecom.png" width="600">

Unlike with memory-based collaborative filters which employ the weighted averaging of product ratings (explicit or implied) between similar users, model-based collaborative filters leverage the features associated with user-product combinations to predict that a given user will click-on or purchase a particular item.  To build such a model, we will need information about users and the products they have purchased.

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
