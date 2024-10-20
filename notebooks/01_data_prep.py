# Databricks notebook source
# MAGIC %md 
# MAGIC # Model Fine Tuning Demo 
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model 
# MAGIC
# MAGIC Generate synthetic question/answer data about Capital Requirements Regulation and use this data to fine tune the Llama 3.0 8B model.
# MAGIC
# MAGIC ## Notebook 1: raw data preparation
# MAGIC
# MAGIC Create the `splitted_documents` table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the Unity Catalog Schema and Catalog location

# COMMAND ----------

# DBTITLE 1,Set the default schema and catalog
dbutils.widgets.text("unity_catalog", "main", "Unity Catalog")
dbutils.widgets.text("unity_schema", "euroreg", "Unity Schema")
unity_catalog = dbutils.widgets.get("unity_catalog")
unity_schema = dbutils.widgets.get("unity_schema")

print("set the Unity Catalog Schema and Catalog using the selection box widgets above")

print(f"Unity Catalog: {unity_catalog}, Unity Schema: {unity_schema} ")
#spark.sql(f"USE {unity_catalog}.{unity_schema}")

# COMMAND ----------

uc_target_catalog = dbutils.widgets.get("unity_catalog")
uc_target_schema = dbutils.widgets.get("unity_schema")
uc_volume_path = f"/Volumes/{uc_target_catalog}/{uc_target_schema}/data"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation and Load Libraries

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# try just re-running this block if there is an issue "None of PyTorch, TensorFlow >= 2.0, or Flax have been found"
import pathlib
import shutil

from databricks.sdk import WorkspaceClient

from finreganalytics.dataprep.dataloading import load_and_clean_data, split
from finreganalytics.utils import get_user_name, set_or_create_catalog_and_database

w = WorkspaceClient()

if (locals().get("uc_target_catalog") is None
        or locals().get("uc_target_schema") is None
        or locals().get("uc_volume_path") is None):
    uc_target_catalog = get_user_name()
    uc_target_schema = get_user_name()
    uc_volume_path = f"/Volumes/{uc_target_catalog}/{uc_target_schema}/data"
    set_or_create_catalog_and_database(uc_target_catalog, uc_target_schema)

    workspace_data_path = str((pathlib.Path.cwd() / ".." / "data").resolve())
    try:
        shutil.copytree(workspace_data_path, uc_volume_path, dirs_exist_ok=True)
    except Exception as e:
        print(e)
    w.dbutils.fs.ls(uc_volume_path)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Ingest the PDF files and  parse their content

# COMMAND ----------

docs_df = load_and_clean_data(uc_volume_path)
display(docs_df) # noqa

# COMMAND ----------

# MAGIC %md After ingesting pdfs and transforming them tot he simple text, we will split the documents and store the chunks as a delta table

# COMMAND ----------

# for testing
docs_short_df = docs_df.limit(1)

# COMMAND ----------

print(type(docs_df))
print(type(docs_short_df))

# COMMAND ----------

splitted_df = split(
    docs_df, hf_tokenizer_name="hf-internal-testing/llama-tokenizer", chunk_size=500
)
display(splitted_df) # noqa

# COMMAND ----------

# MAGIC %md Now let's store the chunks as a delta table

# COMMAND ----------

splitted_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.splitted_documents")
