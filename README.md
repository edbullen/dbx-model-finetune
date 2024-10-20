# MLflow LLM fine-tuning with synthetic data

Training demo for Databricks + MLflow with LLama 3.0 LLM.

1. Load PDFs - European Financial Regulation PDFs: parse + split the PDF data and save to a Delta table `splitted_documents`
2. Generate good quality Q&A data about Capital Requirements Regulation using Chain-of-Thought reasoning.  Write the training and validation data to Delta tables: `qa_dataset_train`, `qa_dataset_val`
3. Fine-tune a Llama 3.0 8B model on the generated synthetic dataset from step 2.  Source the LLM model from the available [Databricks Foundation Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) 
4. Evaluate the model against the eval dataset.  

### Details

In Notebook 2 the function `build_instruction_eval_dataset()` generates an evaluation dataset containing Question, Answer and Context records using supplied LLM
This function links multiple prompt templates into a Langchain chain which applies an LLM to the input Spark dataframe.

The training and validation data-set has a single column, `messages` with a format like this for each row:
```
0:
 {"content": "You are a Regulatory Reporting Assistant.
Please answer the question as precise as possible using information in context.
If you do not know, just say I don't know. ", 
   "role": "system"
} 

1: 
{"content": "Context:
     This Regulation Directive 2006/48/EC Directive 2006/49/EC Article 194(6) Article 92(5) Article 194(7) Article 92(6) Article 194(8) Annex VIII, Part 2, point 1 Article 194(9) Annex VIII, Part 2, point 2 Article 194(10) Article 195 Annex VIII, Part 1, points 3 and 4 Article 196 Annex VIII, Part 1, point 5 Article 197(1) Annex VIII, Part 1, point 7 Article 197(2) Annex VIII, Part 1, point 7 Article 197(3) Annex VIII, Part 1, point 7 Article 197(4) Annex VIII, Part 1, point 8

Please answer the user question using the given context:

What is the relationship between Article 194(6) of the Capital Requirements Regulation (CRR) and the provisions outlined in Directive 2006/48/EC and Directive 2006/49/EC, specifically in regards to the calculation of own funds requirements for institutions?",
   "role": "user"   
 }

2: 
{"content": "Article 194(6) states that \"Institutions shall calculate their own funds requirements in accordance with the provisions laid down in Directive 2006/48/EC and Directive 2006/49/EC, as specified in Article 92(5)\". This means that institutions must follow the guidelines set out in these Directives, which provide detailed rules for calculating own funds requirements.

Specifically, Article 92(5) of the Capital Requirements Regulation (CRR) cross-references the provisions of Directive 2006/48/EC and Directive 2006/49/EC, which outline the requirements for calculating own funds. These Directives specify the elements that institutions must consider when calculating their own funds requirements, including the types of capital that can be included, deductions and adjustments that must be made, and the treatment of specific types of financial instruments


In particular, Directive 2006/48/EC sets out the general framework for calculating own funds, including the definition of own funds and the types of capital that can be included. Directive 2006/49/EC provides additional guidance on the calculation of own funds, including the treatment of specific risk types and the use of internal models.

When calculating their own funds requirements, institutions must carefully review these Directives and ensure that they are complying with all relevant provisions. This includes considering any applicable exemptions or exclusions, as well as ensuring that all necessary deductions and adjustments are made.

In addition to the provisions outlined in the Directives, institutions must also comply with any additional requirements or guidelines set out in Article 92(5) of the CRR. This may include specific requirements for certain types of institutions or financial instruments, as well as guidelines for the use of internal models or other risk measurement techniques.

Overall, the relationship between Article 194(6) of the CRR and the provisions outlined in Directive 2006/48/EC and Directive 2006/49/EC is one of cross-reference and implementation. Institutions must carefully review these Directives and ensure that they are complying with all relevant provisions when calculating their own funds requirements.",
   "role": "assistant"
}        

```

## Setup Notes

This Demo relies upon [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html) with [Serverless Compute](https://docs.databricks.com/en/compute/serverless/index.html).  

Databricks charges for fine-tuning LLMs are listed [here](https://www.databricks.com/product/pricing/mosaic-foundation-model-training)

Check the [regional availability](https://docs.databricks.com/en/resources/feature-region-support.html) for Model Training.

### Data Staging

The source PDFs (located in the `data` folder of this Repo) need to be staged in a Unity Catalog volume:   
`/<catalog>/<schema>/data`
 
