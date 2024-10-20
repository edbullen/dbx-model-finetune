# Databricks notebook source
# MAGIC %md 
# MAGIC # Model Fine Tuning Demo 
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model 
# MAGIC
# MAGIC Generate synthetic question/answer data about Capital Requirements Regulation and use this data to fine tune the Llama 3.0 8B model.
# MAGIC
# MAGIC ## Notebook 3: fine-tune the Llama 3.0 8B model using the training data
# MAGIC
# MAGIC Synthetic training data generated in Notebook 2 is used to perform *Instruction Fine Tuning* (IFT) on the smaller Llama 3 8B model for a specific task.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the Unity Catalog Schema and Catalog location

# COMMAND ----------

dbutils.widgets.text("unity_catalog", "main", "Unity Catalog")
dbutils.widgets.text("unity_schema", "euroreg", "Unity Schema")
unity_catalog = dbutils.widgets.get("unity_catalog")
unity_schema = dbutils.widgets.get("unity_schema")

print("set the Unity Catalog Schema and Catalog using the selection box widgets above")

print(f"Unity Catalog: {unity_catalog}, Unity Schema: {unity_schema} ")
#spark.sql(f"USE {unity_catalog}.{unity_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Libraries and Helper Functions

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# install the Databricks SDK for working with foundation models
%pip install databricks-genai
dbutils.library.restartPython()

# COMMAND ----------

#%load_ext autoreload
#%autoreload 2


# COMMAND ----------

# Access Foundational Models served from Databricks model serving endpoint
from databricks.model_training import foundation_model as fm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base Model and Fine-Tuning Hyper-Parameters 

# COMMAND ----------

# list of available base models in this Databricks workspace
display(fm.get_models().to_pandas()["name"])

# COMMAND ----------

# helper utils
from finreganalytics.utils import setup_logging, get_dbutils, get_current_cluster_id, get_user_name

setup_logging()

uc_target_catalog = dbutils.widgets.get("unity_catalog")
uc_target_schema = dbutils.widgets.get("unity_schema")

# set base_model to point to the path of the foundational model we are going to fine-tune
supported_models = fm.get_models().to_pandas()["name"].to_list()
get_dbutils().widgets.combobox(
    "base_model", "meta-llama/Meta-Llama-3-8B-Instruct", supported_models, "base_model"
)

# model training hyper-parameters; usually very few epochs ("ep") for LLM fine tune (1->3)
get_dbutils().widgets.text("training_duration", "3ep", "training_duration")
get_dbutils().widgets.text("learning_rate", "1e-4", "learning_rate")
get_dbutils().widgets.text(
    "custom_weights_path",
    "",
    "custom_weights_path",
)

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")
custom_weights_path = get_dbutils().widgets.get("custom_weights_path")
if len(custom_weights_path) < 1:
    custom_weights_path = None
cluster_id = get_current_cluster_id()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-Tune the Base Model
# MAGIC
# MAGIC Use the Foundation Model referenced by `base_model` and create a fine-tuned model registered in Unity Catalog as `{uc_target_catalog}.{uc_target_schema}.fin_reg_model_test`.  
# MAGIC   
# MAGIC Monitor the progress in ML Flow Experiments (a link is provided when the run starts).  Allow 10 minutes for completion on 1-node GPU (g4dn.xlarge).  

# COMMAND ----------

run = fm.create(
    model=base_model,
    train_data_path=f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train",
    eval_data_path=f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val",
    register_to=f"{uc_target_catalog}.{uc_target_schema}.fin_reg_model_test",
    training_duration=training_duration,
    learning_rate=learning_rate,
    task_type="CHAT_COMPLETION",
    data_prep_cluster_id=cluster_id
)

# COMMAND ----------

# MAGIC %md
# MAGIC Monitor the run in the MLflow experiments tab.  When complete, check that the model has been registered in Unity Catalog.

# COMMAND ----------

display(fm.get_events(run))

# COMMAND ----------

# Get the MLflow run name
#run.name

# COMMAND ----------

# Display a list of training runs
#display(fm.list())
