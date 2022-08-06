# Introduction 

Collaborative filters leverage similarities between users to make recommendations:

<img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_collabrecom.png" width="600">

Unlike with memory-based collaborative filters which employ the weighted averaging of product ratings (explicit or implied) between similar users, model-based collaborative filters leverage the features associated with user-product combinations to predict that a given user will click-on or purchase a particular item.  To build such a model, we will need information about users and the products they have purchased.

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

