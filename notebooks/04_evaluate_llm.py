# Databricks notebook source
# MAGIC %md 
# MAGIC # Model Adaptation Demo 
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model 
# MAGIC
# MAGIC In this demo we will generate synthetic question/answer data about Capital Requirements Regulation and after that will use this data to fine tune the Llama 3.0 8B model.
# MAGIC
# MAGIC ## Evaluating a Model
# MAGIC In this notebook we will evaluate the model we have fine-tuned during the previous step.
# MAGIC
# MAGIC First, find the Instruction Fine-Tuned model we created in Notebook 3, then serve it via a Databricks model serving endpoint - served as `finreg_demo` in this example.  
# MAGIC   
# MAGIC Then, use the IFT model to generate some answers (the "predictions" we want to validate).  Create a simple prompt->LLM->output chain, feed in the data and collect the results.  
# MAGIC   
# MAGIC Finally, use `mlflow.evaluate()` to use a different LLM (`endpoints:/databricks-meta-llama-3-1-405b-instruct` from a  Databricks foundation model serving endpoint in this example) to judge the prediction outputs compared to the ground-truth answers in the evaluation data-set (in table `qa_dataset_val`, created in Notebook 2)

# COMMAND ----------

# DBTITLE 1,Install libraries
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Langchain dependancies
from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# COMMAND ----------

# DBTITLE 1,Set the Unity catalog and schema location
dbutils.widgets.text("unity_catalog", "main", "Unity Catalog")
dbutils.widgets.text("unity_schema", "euroreg", "Unity Schema")
unity_catalog = dbutils.widgets.get("unity_catalog")
unity_schema = dbutils.widgets.get("unity_schema")

print("set the Unity Catalog Schema and Catalog using the selection box widgets above")

print(f"Unity Catalog: {unity_catalog}, Unity Schema: {unity_schema} ")
#spark.sql(f"USE {unity_catalog}.{unity_schema}")
uc_target_catalog = dbutils.widgets.get("unity_catalog")
uc_target_schema = dbutils.widgets.get("unity_schema")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Serve the Model to Evaluate via a Serving Endpoint
# MAGIC We will need to create a PT (Provisioned Throughput) endpoint for the model we have fine-tuned during the previous step.    
# MAGIC This can be done manually using Databricks UI - select the model fine-tuned in Notebook 3 that was saved in Unity Catalog.
# MAGIC
# MAGIC
# MAGIC ![model serving endpoint](../doc/create_model_serving_endpoint.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build an processing chain for the LLM based off the eval dataset

# COMMAND ----------

SYSTEM_INSTRUCTION = """You are a Regulatory Reporting Assistant.
Please answer the question as precise as possible using information in context.
If you do not know, just say I don't know. """


def build_retrievalqa_with_context_chain(llm: BaseLanguageModel):
  """Build a RetrievalQA chain with context"""
  prompt = ChatPromptTemplate.from_messages([(
    "system", SYSTEM_INSTRUCTION),
    ("user", """Context:\n {context}\n\n Please answer the user question using the given context:\n {question}"""),])

  chain = prompt | llm | StrOutputParser()

  return chain

# COMMAND ----------

llm = ChatDatabricks(endpoint="finreg_demo", temperature=0.99)
qa_chain_with_ctx = build_retrievalqa_with_context_chain(llm)

# COMMAND ----------

# get the eval data-set that was created in Notebook 2 as a Pandas DF
eval_df = spark.read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset_val").toPandas()

# COMMAND ----------

eval_df.head(10)

# COMMAND ----------

# Define function to run a supplied chain to generate answers for a list of questions - this will be the prediction to evaluate
from typing import List, Dict

def run_chain_for_eval_data(chain, input_prompts: List[Dict[str, str]]) -> List[str]:
    """
    Generates answers for the list of questions using defined LCEL chain
    :param chain: chain to use for inference
    :param input_prompts: list of questions
    :return: list of resulting strings
    """
    return chain.with_retry(
        stop_after_attempt=100, wait_exponential_jitter=False
    ).batch(input_prompts, config={"max_concurrency": 4})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Chain against the Validation set
# MAGIC
# MAGIC This adds the output answer from the fine-tuned LLM as column `prediction`.  The original `question` and `answer` columns are retained so the ground-truth `answer` can be compared with our LLM's predicted answer (`prediction`)

# COMMAND ----------

# add the prediction col to the eval_df
eval_df["prediction"] = run_chain_for_eval_data(
        qa_chain_with_ctx, eval_df[["context", "question", "answer"]].to_dict(orient="records")
    )

# COMMAND ----------

# now we have a question, answer (ground truth), prediction (from the fine-tune model) to evaluate
display(eval_df[["question", "answer", "prediction"]].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use an LLM-as-a-Judge to evaluate the model predictions
# MAGIC
# MAGIC See the [MLflow documentation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#metrics-with-llm-as-the-judge) for details of the metrics MLflow can provide using an LLM-as-a-Judge evaluation method.

# COMMAND ----------

# count of how many rows of evaluation data there is
eval_df.count()

# COMMAND ----------

import mlflow

# use one of the available foundational models being served by Databricks - in this case databricks-meta-llama-3-1-405b-instruct (check model serving console to see what is available)

# MLflow evaluate needs the "question" column to be named "inputs"
eval_df = eval_df.reset_index(drop=True).rename(columns={"question": "inputs"})

eval_results = mlflow.evaluate(
            data=eval_df,
            targets="answer",
            predictions="prediction",
            extra_metrics=[
                mlflow.metrics.genai.answer_similarity(model=f"endpoints:/databricks-meta-llama-3-1-405b-instruct")
            ],
            evaluators="default",
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpreting the `answer_similarity` metric
# MAGIC
# MAGIC The range of possible scores for the answer_similarity metric returned by mlflow.metrics.genai.answer_similarity is from 1 to 5:   
# MAGIC
# MAGIC + Score 1: The output has little to no semantic similarity to the provided targets.  
# MAGIC + Score 2: The output displays partial semantic similarity to the provided targets on some aspects.  
# MAGIC + Score 3: The output has moderate semantic similarity to the provided targets.  
# MAGIC + Score 4: The output aligns with the provided targets in most aspects and has substantial semantic similarity.  
# MAGIC + Score 5: The output closely aligns with the provided targets in all significant aspects.  

# COMMAND ----------

for m in eval_results.metrics:
  print(m, eval_results.metrics[m])

# COMMAND ----------

# Print per-data evaluation results
display(eval_results.tables['eval_results_table'])
