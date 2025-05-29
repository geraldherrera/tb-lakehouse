# Databricks notebook source
from pyspark.sql.functions import expr

# COMMAND ----------

# Load the base dataset (Bakehouse customer reviews)
df = spark.table("samples.bakehouse.media_customer_reviews")

# COMMAND ----------

# Apply the ai_query() LLM function using SQL expression inside PySpark
# This will classify each review as 'positive' or 'negative'
df_with_sentiment = df.select(
    "review",
    expr("""ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        "Evaluate if the review is positive or negative. Only answer with 'positive' or 'negative'" || review
    ) AS sentiment""")
)

# COMMAND ----------

# Display 10 rows of results (review + sentiment prediction)
df_with_sentiment.show(10, truncate=False)