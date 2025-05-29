# Databricks notebook source
# MAGIC %md
# MAGIC # Code to test the SCD2 in Silver

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

# COMMAND ----------

# Set catalog and schema
spark.sql("USE CATALOG dbw_dataplatform_dev_ghe_1022068885507136")
spark.sql("USE SCHEMA bronze")

# COMMAND ----------

# Simulate an UPDATE in source
spark.sql("SELECT * FROM address WHERE City = 'Bothell'").show()
spark.sql("""
UPDATE address
SET PostalCode = '12345', ModifiedDate = current_timestamp()
WHERE City = 'Bothell'
""")

# COMMAND ----------

# Simulate a DELETE in source
spark.sql("SELECT * FROM address WHERE City = 'Surrey'").show()
spark.sql("DELETE FROM address WHERE City = 'Surrey'")

# COMMAND ----------

# Simulate an INSERT in source
spark.sql("SELECT * FROM bronze.Address ORDER BY AddressID DESC").show()

# COMMAND ----------

# Simulate INSERT+DELETE via PK modification
spark.sql("""
UPDATE bronze.Address
SET AddressID = 11383
WHERE AddressID = 1105
""")

# COMMAND ----------

# Run ETL externally before continuing this test

# COMMAND ----------

# Check results in Silver after ETL execution
spark.sql("USE SCHEMA silver")
spark.sql("SELECT * FROM address WHERE city = 'Bothell' ORDER BY address_id, _tf_valid_from").show()
spark.sql("SELECT * FROM address WHERE city = 'Surrey' ORDER BY address_id, _tf_valid_from").show()
spark.sql("SELECT * FROM address WHERE address_id IN (11383, 1105)").show()