# Databricks notebook source
# MAGIC %md 
# MAGIC # Model Fine Tuning Demo 
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model 
# MAGIC
# MAGIC Generate synthetic question/answer data about Capital Requirements Regulation and use this data to fine tune the Llama 3.0 8B model.
# MAGIC
# MAGIC ## Notebook 2: synthetic training data generation using Llama 3.1 70B
# MAGIC
# MAGIC create the tables:
# MAGIC - `qa_dataset`
# MAGIC - `qa_dataset_train`
# MAGIC - `qa_dataset_val`
# MAGIC - `qa_instructions_train` - synthetic training data-set for fine-tuning the LLM
# MAGIC - `qa_instructions_val`  - synthetic validation data-set for testing the LLM

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic Data Generation
# MAGIC In this notebook we will use the CoT (Chain of Thought) technique to create high quality questions and answers about Capital Requirements Regulation.
# MAGIC We will iterate over all the chunks we created in the first step and generate a question about the facts mentioned in the chunk and then ask an LLM to answer this question using the provided chunk.  
# MAGIC
# MAGIC This synthetic data will be used by *Notebook 3* to fine-tune an LLM.

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

# MAGIC %md
# MAGIC ## Load Libraries and Helper Functions

# COMMAND ----------

# DBTITLE 1,Install Libraries
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#%pip install mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

uc_target_catalog = dbutils.widgets.get("unity_catalog")
uc_target_schema = dbutils.widgets.get("unity_schema")
uc_volume_path = f"/Volumes/{uc_target_catalog}/{uc_target_schema}/data"

# COMMAND ----------

# DBTITLE 1,import helper functions to prepare the dataset
# ChatDatabricks class wraps a chat model endpoint hosted on Databricks Model Serving. 
from langchain_community.chat_models.databricks import ChatDatabricks
from pyspark.sql.functions import rand

# "prepare_ift_dataset" applys the UDF "transform_chat_udf" to a set schema in a Spark DF
from finreganalytics.dataprep.ift_data_prep import (
    prepare_ift_dataset,
)

# build_instruction_eval_dataset generates an evaluation dataset containing Question, Answer and Context records using supplied LLM
from finreganalytics.dataprep.qagen import build_instruction_eval_dataset
from finreganalytics.utils import get_spark, get_user_name, batchify

# COMMAND ----------

# MAGIC %md 
# MAGIC ## LLM Chat Templates - Question Templates
# MAGIC Step 1: define the prompt to generate an initial question that corresponds to the chunk of text.

# COMMAND ----------

INITIAL_QUESTION_GENERATION_PROMPT_TMPL = """\
Context information is below.

---------------------
{context}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. 
Your task is to come up with a question for an upcoming  quiz/examination on Capital Requirements Regulation (CRR). 
The questions should be diverse in nature across the document. Restrict the questions to the
context information provided.
Please generate exactly one questions and no more.
Do not include any further information.

"""

# COMMAND ----------

# MAGIC %md
# MAGIC Step 2: For the second prompt template, we will use CoT and ask an LLM to give reasons and thoughts behind the choice of this particlar question.  
# MAGIC We will also ask the LLM to provide some judgment as to whether this is a good question and also to provide some ideas for improvement. 

# COMMAND ----------

JUDGEMENT_QUESTION_GENERATION_PROMPT_TMPL = """\

Context information:
---------------------
{context}
---------------------

Question
---------------------
{question}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. 
Your task is to make a decision if the mentioned above question is good for an upcoming  quiz/examination on Capital Requirements Regulation (CRR). 
Please come up with some thoughts first about the question above, think if this is a good one for students. 
After that make a decision and explain why you think it is a good or bad one.

"""

# COMMAND ----------

# MAGIC %md
# MAGIC Step 3: Finally we create a prompt to ask the LLM to improve the generated question using the thoughts and improvement ideas we generated in the previous cell.

# COMMAND ----------

IMPROVE_QUESTION_GENERATION_PROMPT_TMPL = """\
Context information:
---------------------
{context}
---------------------

Question
---------------------
{question}
---------------------

Judgement
---------------------
{judgement}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor in Financial Regulation. 
Your task is to come up with a question for an upcoming  quiz/examination on Capital Requirements Regulation (CRR). 
The questions should be diverse in nature across the document. Restrict the questions to the
context information provided.
Above you have a question and a judgement about its quality.
Improve the question according to the judgement and rewrite it to address points indicated in the judgement. 
If the question is already perfect just output it without any modifications.
Do not include any further information and do not write if the question is good or bad or what you have modified.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Chat Templates - Answer Templates
# MAGIC
# MAGIC Now we have defined the question templates, next we will define 3 prompts to generate an answer using the same logic:
# MAGIC - Generate initial answer
# MAGIC - Generate thoughts and reasoning behind it, and then come up with some judgments and ideas for improvement.
# MAGIC - Use these ideas to improve the answer.

# COMMAND ----------

INITIAL_ANSWER_GENERATION_PROMPT_TMPL = """
Context information:
---------------------
{context}
---------------------

You are an expert in European Financial Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
If the question is not related to one of these topics, kindly decline to answer. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible.
Please do not repeat the answer and do not add any additional information. 
Please answer the question above using information given in context.

Question: {question}

Answer:
"""

JUDGEMENT_ANSWER_GENERATION_PROMPT_TMPL = """
Context information:
---------------------
{context}
---------------------

Question
---------------------
{question}
---------------------

Answer
---------------------
{answer}
---------------------

You are an expert in European Financial Regulation and Capital Requirements Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
Your task is to make a decision if the mentioned above answer fully and correctly answers the question mentioned above. 
Please come up with some thoughts first about the answer above, think if this is a correct and full answer. 
After that make a decision and explain why you think it is a good or bad one.

Question: {question}

Answer:
"""

IMPROVE_ANSWER_GENERATION_PROMPT_TMPL = """
Context information:
---------------------
{context}
---------------------

Question
---------------------
{question}
---------------------

Answer
---------------------
{answer}
---------------------

Judgement
---------------------
{judgement}
---------------------

You are an expert in European Financial Regulation and Capital Requirements Regulation. 
You are answering questions related to Financial Regulation for the Financial Institutes in the European Union. 
Your task is to improve the mentioned above answer using the provided above judgement and rewrite it to address points indicated in the judgement. 
If the answer is already perfect just output it without any modifications.
Do not include any further information and do not write if the question is good or bad or what you have modified.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Chat Prompt to process the raw data and generate training data
# MAGIC Now we will use Llama 3.1 70B to run all these prompts for each chunk. 
# MAGIC We will pass them to the `build_instruction_eval_dataset` function (imported earlier from `finreganalytics.dataprep.qagen`).  
# MAGIC - This function will iterate over the input chunks of data, build the final prompts and send them to Llama 3.1 70B.  
# MAGIC - The output data is stored in table `qa_dataset`
# MAGIC
# MAGIC The Langchain chain defined in `build_instruction_eval_dataset` processes the raw chunks of text data in `splitted_documents` (converted to Pandas) to produce synthetic model training example questions and answers.
# MAGIC
# MAGIC ![question and answer chain](../doc/langchain_questions_and_answers_chain.png)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC In the next cell, we will generate just two questions to validate our approach.

# COMMAND ----------

# DBTITLE 1,Test the questions and answers chain
chunks_df = get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.splitted_documents").orderBy(rand())
chunks = chunks_df.toPandas()["text"].values.tolist()[:100]

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", temperature=0.9)

# The LangChain "chain" is defined in finreganalytics.qagen.build_instruction_eval_dataset
qa_questions_df = build_instruction_eval_dataset(
    chunks[:2],
    llm,
    initial_question_prompt_template_str=INITIAL_QUESTION_GENERATION_PROMPT_TMPL,
    judgement_question_prompt_template_str=JUDGEMENT_QUESTION_GENERATION_PROMPT_TMPL,
    improve_question_prompt_template_str=IMPROVE_QUESTION_GENERATION_PROMPT_TMPL,
    initial_answer_prompt_template_str=INITIAL_ANSWER_GENERATION_PROMPT_TMPL,
    judgement_answer_prompt_template_str=JUDGEMENT_ANSWER_GENERATION_PROMPT_TMPL,
    improve_answer_prompt_template_str=IMPROVE_ANSWER_GENERATION_PROMPT_TMPL,
)
display(qa_questions_df)  # noqa

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have validated the entire approach, we can run it iteratively over the entire dataset. We will generate multiple questions for each chunk and iterate over the entire dataset in chunks of 200 chunks. We will store the generated questions and answers for each chunk independently.
# MAGIC
# MAGIC This takes about 1hr to run on a single gx4dn GPU.

# COMMAND ----------

# DBTITLE 1,Run the Question and Answer Generator
number_of_questions = 2
chunk_length = 200

for i in range(number_of_questions):
    print(f"Iteration: {i}\n")
    for current_chunk in batchify(chunks, chunk_length):
        print(f"Chunk length: {len(current_chunk)}\n")
        qa_questions_df = build_instruction_eval_dataset(
            current_chunk,
            llm,
            initial_question_prompt_template_str=INITIAL_QUESTION_GENERATION_PROMPT_TMPL,
            judgement_question_prompt_template_str=JUDGEMENT_QUESTION_GENERATION_PROMPT_TMPL,
            improve_question_prompt_template_str=IMPROVE_QUESTION_GENERATION_PROMPT_TMPL,
            initial_answer_prompt_template_str=INITIAL_ANSWER_GENERATION_PROMPT_TMPL,
            judgement_answer_prompt_template_str=JUDGEMENT_ANSWER_GENERATION_PROMPT_TMPL,
            improve_answer_prompt_template_str=IMPROVE_ANSWER_GENERATION_PROMPT_TMPL,
        )
        (
            get_spark()
            .createDataFrame(qa_questions_df)
            .write
            .mode("append")
            .saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset")
        )

# COMMAND ----------

display(get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Generate Chat Completions 
# MAGIC Now we should have all questions and answers ready and we can transform them to the instruction dataset formatted as chat completions. We will use `prepare_ift_dataset` (imported earlier from `finreganalytics.dataprep.ift_data_prep`).  
# MAGIC    
# MAGIC The `prepare_ift_dataset` function applies the `transform_chat_udf` to the dataframe which executes the `format_chat_completion` function as a parallel spark process (These functions are also defined in `ift_data_prep.py`).  
# MAGIC
# MAGIC - Data in `qa_dataset` is read in
# MAGIC - This is split into train and validate `qa_dataset_train` and `qa_dataset_val`
# MAGIC - The output dataset is `qa_instructions_train` and `qa_instructions_val`

# COMMAND ----------

qa_train_df, qa_val_df = get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset").orderBy(
    rand()).randomSplit([0.9, 0.1])
qa_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset_train")
qa_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset_val")

qa_ift_train_df = prepare_ift_dataset(qa_train_df, limit=-1)
qa_ift_val_df = prepare_ift_dataset(qa_val_df, limit=-1)

qa_ift_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train")
qa_ift_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val")

# COMMAND ----------

# MAGIC %md
# MAGIC **QA Dataset Sample**
# MAGIC
# MAGIC A single column `messages` has rows with list-arrays `[{"content":...}, {"content":...}, {"content":...}]`.   
# MAGIC EG: 
# MAGIC
# MAGIC 0: {"content": "You are a Regulatory Reporting Assistant.\nPlease answer the question as precise as possible using information in context.\nIf you do not know, just say I don't know. ", "role": "system"}
# MAGIC
# MAGIC
# MAGIC 1: {"content": "Context:\n (ii) from the amount in point (i):(i) the sum of the total value of securities denominated in that currency lent, sold orprovided under the master netting agreement and the amount of cash in thatcurrency lent or transferred under that agreement;(ii) the sum of the total value of securitie s denominated in that currency borrow ed,purchased or received under the master netting agreement and the amount of cash inthat currency borrowed or received under that agreement;(c) apply the volatility adjustment approp riate to a given group of securities or to a cashposition to the absolute value of the positive or negative net position in the securities in thatgroup;(d) apply the foreign exchange risk\n\n Please answer the user question using the given context:\n What is the purpose of calculating the sum of the total value of securities denominated in that currency lent, sold or provided under the master netting agreement, as mentioned in point (i), in the context of determining capital requirements?", "role": "user"}
# MAGIC
# MAGIC
# MAGIC 2: {"content": "The purpose of calculating the sum of the total value of securities denominated in that currency lent, sold or provided under the master netting agreement, as mentioned in point (i), is to determine the total value of securities and cash lent or provided under the agreement. This calculation is part of a broader process of calculating potential future exposure to credit risk, which is a key component of determining capital requirements. Specifically, the calculation in point (i) is used to determine the gross exposure under the master netting agreement, which is then compared to the gross exposure under the agreement as calculated in point (ii). The resulting net position is then used to apply the volatility adjustment, and subsequently to determine the capital requirement. By calculating the net position in securities in a given currency, the institution can assess its potential exposure to credit risk and determine the necessary capital requirements to cover this exposure.", "role": "assistant"}
# MAGIC

# COMMAND ----------

display(get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val"))  

# COMMAND ----------

display(get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train")) 

# COMMAND ----------


