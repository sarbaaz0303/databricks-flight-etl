# Databricks notebook source
# MAGIC %md
# MAGIC # Efficient Ingestion and Preprocessing of Large-Scale Flight Data with Apache Spark on Databricks
# MAGIC
# MAGIC This notebook demonstrates a robust and modular approach to ingesting, cleaning, and validating large-scale flight data using Apache Spark. The data pre-processing follows the 6 dimensions of data quality:
# MAGIC
# MAGIC ### Data Quality Dimensions
# MAGIC ![Data Quality Dimensions](https://camo.githubusercontent.com/777e7d74c5682fbd848b7b0d4a0bbaf97576c2285f65975a8c492e82e4d47e77/68747470733a2f2f7777772e7061636966696364617461696e7465677261746f72732e636f6d2f68756266732f446174612d7175616c6974792d64696d656e73696f6e732e6a7067)
# MAGIC
# MAGIC **Key Aspects of Data Quality:**
# MAGIC - **Accuracy**: Ensuring data is correct and reliable.
# MAGIC - **Completeness**: No missing values or critical gaps.
# MAGIC - **Consistency**: Data remains uniform across different sources.
# MAGIC - **Timeliness**: Data is up-to-date and available when needed.
# MAGIC - **Validity**: Data adheres to predefined formats and constraints.
# MAGIC - **Uniqueness**: No duplicate records exist.
# MAGIC - **Integrity**: Data relationships are maintained correctly.
# MAGIC
# MAGIC High-quality data is essential for analytics, decision-making, and operational efficiency.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setting Up Environment & Downloading Data
# MAGIC
# MAGIC We start by creating a data directory and fetching both the metadata and flight data files using shell commands.

# COMMAND ----------

# MAGIC %sh
# MAGIC # Clear the data directory
# MAGIC rm -rf /dbfs/data

# COMMAND ----------

# MAGIC %sh
# MAGIC # Create the data directory if it doesn't exist
# MAGIC mkdir -p /dbfs/data
# MAGIC
# MAGIC # Download metadata and flight data (2004-2008)
# MAGIC wget "https://dataverse.harvard.edu/api/datasets/export?exporter=dataverse_json&persistentId=doi%3A10.7910/DVN/HG7NV7" -O /dbfs/data/metadata.json
# MAGIC
# MAGIC wget "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HG7NV7/CCAZGT" -O /dbfs/data/2004.csv.bz2
# MAGIC wget "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HG7NV7/JTFT25" -O /dbfs/data/2005.csv.bz2
# MAGIC wget "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HG7NV7/EPIFFT" -O /dbfs/data/2006.csv.bz2
# MAGIC wget "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HG7NV7/2BHLWK" -O /dbfs/data/2007.csv.bz2
# MAGIC wget "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HG7NV7/EIR0RA" -O /dbfs/data/2008.csv.bz2

# COMMAND ----------

# List the files in /data directory
display(dbutils.fs.ls("/data"))

# COMMAND ----------

# MAGIC %sh
# MAGIC # Unzip the flight data files
# MAGIC for file in /dbfs/data/*.bz2; do 
# MAGIC     if [[ -f $file ]]; then 
# MAGIC         bzip2 -d "$file" && echo "Unzipped: $file"; 
# MAGIC     else 
# MAGIC         echo "No .bz2 files found in the directory."; 
# MAGIC     fi 
# MAGIC done 

# COMMAND ----------

display(dbutils.fs.ls("/data"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Ingestion and Initial Exploration
# MAGIC
# MAGIC Here we load the metadata and flight CSV files into Spark DataFrames and define a schema for the flight data.

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
import matplotlib.pyplot as plt

# Load metadata for reference
flights_metadata = spark.read.json("/data/metadata.json")
display(flights_metadata)

# COMMAND ----------

# Define schema for flight data
schema = StructType([
    StructField('Year', IntegerType()),
    StructField('Month', IntegerType()),
    StructField('DayofMonth', IntegerType()),
    StructField('DayOfWeek', IntegerType()),
    StructField('DepTime', IntegerType()),
    StructField('CRSDepTime', IntegerType()),
    StructField('ArrTime', IntegerType()),
    StructField('CRSArrTime', IntegerType()),
    StructField('UniqueCarrier', StringType()),
    StructField('FlightNum', IntegerType()),
    StructField('TailNum', IntegerType()),
    StructField('ActualElapsedTime', IntegerType()),
    StructField('CRSElapsedTime', IntegerType()),
    StructField('AirTime', IntegerType()),
    StructField('ArrDelay', IntegerType()),
    StructField('DepDelay', IntegerType()),
    StructField('Origin', StringType()),
    StructField('Dest', StringType()),
    StructField('Distance', IntegerType()),
    StructField('TaxiIn', IntegerType()),
    StructField('TaxiOut', IntegerType()),
    StructField('Cancelled', IntegerType()),
    StructField('CancellationCode', StringType()),
    StructField('Diverted', IntegerType()),
    StructField('CarrierDelay', IntegerType()),
    StructField('WeatherDelay', IntegerType()),
    StructField('NASDelay', IntegerType()),
    StructField('SecurityDelay', IntegerType()),
    StructField('LateAircraftDelay', IntegerType())
])

# COMMAND ----------

# Read one of the yearly files to preview the data (e.g., 2004)
flights_2004 = spark.read.csv("/data/2004.csv", header=True, schema=schema)
flights_2004.limit(5).display()

# COMMAND ----------

flights_2004.printSchema()

# COMMAND ----------

flights_2004.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Consolidation and Cleaning
# MAGIC
# MAGIC In this section we load all the yearly files and consolidate them into a single DataFrame. We also show an example of nullifying columns that may be problematic and later drop columns with no data.

# COMMAND ----------

# Load other years
flights_2005 = spark.read.csv("/data/2005.csv", header=True, schema=schema)
flights_2006 = spark.read.csv("/data/2006.csv", header=True, schema=schema)
flights_2007 = spark.read.csv("/data/2007.csv", header=True, schema=schema)
flights_2008 = spark.read.csv("/data/2008.csv", header=True, schema=schema)

# Combine data if needed. Here we use 2008 for demonstration.
df_flights = flights_2004.union(flights_2005).union(flights_2006).union(flights_2007).union(flights_2008)
df_flights = flights_2008

df_flights.limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Data Completeness and Column Validation
# MAGIC
# MAGIC We check for columns with no data and drop them. We also evaluate missing values and then drop any rows with nulls.

# COMMAND ----------

# Calculate non-null counts for each column
non_null_counts_df = df_flights.agg(*[
    sum(col(column).isNotNull().cast("int")).alias(column) for column in df_flights.columns
])
non_null_counts = non_null_counts_df.first().asDict()
columns_to_drop = [col_name for col_name, count in non_null_counts.items() if count == 0]

# Drop columns with no data and log the dropped columns
main_df = df_flights.drop(*columns_to_drop)
for col_name in columns_to_drop:
    print(f"Dropped column: {col_name}")

# COMMAND ----------

display(main_df.describe())

# COMMAND ----------

# Compute null counts per column (as both absolute and percentage)
total_count = main_df.count()
null_counts_df = main_df.agg(*[
    sum(col(c).isNull().cast("int")).alias(c) for c in main_df.columns
])
null_counts = null_counts_df.first().asDict()
with_nulls = [
    {col_name: [null_count, null_count / total_count]} 
    for col_name, null_count in null_counts.items() if null_count > 0
]
print("Columns with null counts (absolute and ratio):", with_nulls)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Accuracy and Visualization
# MAGIC
# MAGIC We generate histograms for numerical (integer) columns to check their distributions.

# COMMAND ----------

# Plot histograms for integer columns
from pyspark.sql.types import IntegerType

for column in main_df.columns:
    if isinstance(main_df.schema[column].dataType, IntegerType):
        pd_df = main_df.select(col(column)).toPandas()
        if not pd_df.empty:
            pd_df.hist(column=column, bins=20, edgecolor='black')
            plt.title(f"Histogram for {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Consistency Checks
# MAGIC
# MAGIC We analyze string columns to check that categorical data remains consistent.

# COMMAND ----------

string_columns = [c for c, t in main_df.dtypes if t == "string"]
print("String columns in the data:", string_columns)

for column in string_columns:
    main_df.groupBy(column).count().select(column, "count").orderBy(column).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Uniqueness
# MAGIC
# MAGIC We check for duplicate rows and duplicates based on a compound primary key. For the compound key, we generate a timestamp from the departure time.
# MAGIC
# MAGIC **Compound Primary Key Fields:**
# MAGIC - Origin
# MAGIC - UniqueCarrier
# MAGIC - FlightNum
# MAGIC - DepTime_Timestamp

# COMMAND ----------

# Count identical rows
duplicate_count = main_df.groupBy(main_df.columns).count().where(col("count") > 1).count()
print(f"Number of identical duplicate rows: {duplicate_count}")

# Remove identical duplicate rows
unique_rows = main_df.dropDuplicates()
duplicate_after_drop = unique_rows.groupBy(unique_rows.columns).count().where(col("count") > 1).count()
print(f"Duplicate rows after dropDuplicates: {duplicate_after_drop}")

# Create departure timestamp and check duplicates based on the compound key
primary_key_fields = ["Origin", "UniqueCarrier", "FlightNum", "DepTime_Timestamp"]

unique_rows.createOrReplaceTempView("unique_rows")
unique_rows = spark.sql("""
SELECT *,
  CASE 
    WHEN substring(cast(DepTime AS string), 1, 2) = '24' THEN '00'
    WHEN length(cast(DepTime AS string)) < 3 THEN '00'
    WHEN length(cast(DepTime AS string)) = 3 THEN substring(cast(DepTime AS string), 1, 1)
    ELSE substring(cast(DepTime AS string), 1, 2)
  END AS DepTime_Hour,
  
  right(cast(DepTime AS string), 2) AS DepTime_Min,
  
  to_timestamp(
    concat(
      cast(Year AS string), '-', 
      lpad(cast(Month AS string), 2, '0'), '-', 
      lpad(cast(DayofMonth AS string), 2, '0'), ' ',
      lpad(
        CASE 
          WHEN substring(cast(DepTime AS string), 1, 2) = '24' THEN '00'
          WHEN length(cast(DepTime AS string)) < 3 THEN '00'
          WHEN length(cast(DepTime AS string)) = 3 THEN substring(cast(DepTime AS string), 1, 1)
          ELSE substring(cast(DepTime AS string), 1, 2)
        END, 2, '0'
      ),
      ':',
      lpad(right(cast(DepTime AS string), 2), 2, '0')
    ),
    'yyyy-MM-dd HH:mm'
  ) AS DepTime_Timestamp
FROM unique_rows
""")

compound_duplicates = unique_rows.groupBy(primary_key_fields).count().where(col("count") > 1).count()
print(f"Duplicate rows based on the compound primary key: {compound_duplicates}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data Validity Checks
# MAGIC
# MAGIC We perform several validity checks to ensure that numerical values fall within expected ranges.
# MAGIC
# MAGIC **Checks Include:**
# MAGIC - Year < current year
# MAGIC - Month between 1 and 12
# MAGIC - DayofMonth between 1 and 31
# MAGIC - DayOfWeek between 1 and 7
# MAGIC - Time fields between 1 and 2400
# MAGIC - FlightNum and Distance greater than 0

# COMMAND ----------

# Current year for validation
current_year = year(current_date())

def check_validity(df, column, condition, message):
    failures = df.filter(~condition)
    print(f"{message} Failures:")
    if failures.count() > 0:
        failures.show()
    else:
        print("All rows meet the validity criteria\n")

# Check Year
check_validity(unique_rows, "Year", col("Year") < current_year, "Year Check")

# Check Month (1-12)
check_validity(unique_rows, "Month", (col("Month") >= 1) & (col("Month") <= 12), "Month Check")

# Check DayofMonth (1-31)
check_validity(unique_rows, "DayofMonth", (col("DayofMonth") >= 1) & (col("DayofMonth") <= 31), "DayofMonth Check")

# Check DayOfWeek (1-7)
check_validity(unique_rows, "DayOfWeek", (col("DayOfWeek") >= 1) & (col("DayOfWeek") <= 7), "DayOfWeek Check")

# Check DepTime (1-2400)
check_validity(unique_rows, "DepTime", (col("DepTime") >= 1) & (col("DepTime") <= 2400), "DepTime Check")

# Check CRSDepTime (1-2400)
check_validity(unique_rows, "CRSDepTime", (col("CRSDepTime") >= 1) & (col("CRSDepTime") <= 2400), "CRSDepTime Check")

# Check ArrTime (1-2400)
check_validity(unique_rows, "ArrTime", (col("ArrTime") >= 1) & (col("ArrTime") <= 2400), "ArrTime Check")

# Check CRSArrTime (1-2400)
check_validity(unique_rows, "CRSArrTime", (col("CRSArrTime") >= 1) & (col("CRSArrTime") <= 2400), "CRSArrTime Check")

# Check FlightNum > 0
check_validity(unique_rows, "FlightNum", col("FlightNum") > 0, "FlightNum Check")

# Check Distance > 0
check_validity(unique_rows, "Distance", col("Distance") > 0, "Distance Check")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Data Timeliness
# MAGIC
# MAGIC We analyze the flight count per day to determine data timeliness and identify any missing days.

# COMMAND ----------

# Create a timestamp column for departure date and group by it
unique_rows.createOrReplaceTempView("unique_rows")
range_rows = spark.sql(f"""
    SELECT 
    to_timestamp(
        concat(
        cast(Year AS string), '-', 
        lpad(cast(Month AS string), 2, '0'), '-', 
        lpad(cast(DayofMonth AS string), 2, '0')
        ), 
        'yyyy-MM-dd'
    ) AS DepTime_Date,
    *
    FROM unique_rows
""")

range_rows= range_rows.groupBy("DepTime_Date").count().orderBy("DepTime_Date")
range_rows.display()

# COMMAND ----------

# Identify the continuous date range
min_date = range_rows.agg(min("DepTime_Date")).collect()[0][0]
max_date = range_rows.agg(max("DepTime_Date")).collect()[0][0]

# Create a continuous date range DataFrame
date_range = spark.range(0, (max_date - min_date).days + 1).select(
    expr("date_add('{}', cast(id as int)) as days".format(min_date))
)
# Add a placeholder count column
t2 = date_range.withColumn("count", lit(0)).withColumnRenamed("count", "t2_count")
range_rows = range_rows.withColumnRenamed("count", "range_count")

# Left join to identify missing days
result = t2.join(range_rows, t2.days == range_rows.DepTime_Date, "left") \
    .withColumn("count", col("t2_count") + coalesce(col("range_count"), lit(0))) \
    .select(t2["days"], col("count"))
print("Days with no flight data:")
result.where(col("count") == 0).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Storing the Cleaned Data
# MAGIC
# MAGIC Finally, we estimate the size of the cleaned DataFrame, repartition it, and write it as a Parquet file.
# MAGIC
# MAGIC **Note:** Adjust the target year (or path) as needed.

# COMMAND ----------

import math
import builtins
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer

# Estimate the DataFrame size (in MB)
rdd = unique_rows.rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
obj = rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)
size = sc._jvm.org.apache.spark.util.SizeEstimator.estimate(obj)
size_MB = size / 1e6
print(f'The dataframe is approximately {size_MB:.2f} MB')

# Calculate number of partitions (assuming ~200 MB per partition)
partitions = builtins.max(1, math.ceil(size_MB / 200))
print(f"Repartitioning to {partitions} partition(s)")

# COMMAND ----------

# Repartition and write the cleaned data to Parquet
target_year = 2008
output_path = f"/data/preprocessed_flight_data_{target_year}"

coalesced_df = unique_rows.coalesce(partitions)
coalesced_df.write.mode("overwrite").parquet(output_path)
print(f"Data successfully written to {output_path}")

# COMMAND ----------

display(dbutils.fs.ls('/data/preprocessed_flight_data_2008/'))

# COMMAND ----------

spark.read.parquet('/data/preprocessed_flight_data_2008/').display()