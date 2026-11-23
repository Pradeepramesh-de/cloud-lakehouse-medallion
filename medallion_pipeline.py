"""
End-to-End Lakehouse using Medallion Architecture (Bronze -> Silver -> Gold)

Setup:
    pip install pyspark==3.5.1 delta-spark==3.2.0
    (needs Java 8/11/17 installed)

Run:
    python medallion_pipeline.py

To move to the cloud, only change LAKE_PATH:
    AWS   : s3a://your-bucket/lakehouse     (+ hadoop-aws jars and credentials)
    Azure : abfss://container@account.dfs.core.windows.net/lakehouse
"""

import os
import json
import random
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from delta import configure_spark_with_delta_pip

# --------------------------------------------------------------------------
# 0. CONFIG
# Why: one base path means the same code works on local disk, S3 or ADLS.
# --------------------------------------------------------------------------
LAKE_PATH = os.getenv("LAKE_PATH", "./lakehouse")
RAW_PATH = f"{LAKE_PATH}/raw"
BRONZE = f"{LAKE_PATH}/bronze/orders"
SILVER = f"{LAKE_PATH}/silver/orders"
GOLD = f"{LAKE_PATH}/gold"


def get_spark():
    # Why: Delta Lake gives ACID transactions, schema enforcement and
    # time travel on top of plain Parquet files. This is the "lakehouse" part.
    builder = (
        SparkSession.builder.appName("medallion-lakehouse")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


# --------------------------------------------------------------------------
# 1. GENERATE SAMPLE RAW DATA
# Why: simulates an API/batch source. It is deliberately messy (duplicates,
# nulls, bad prices, extra spaces) so Silver has real cleaning work to do.
# --------------------------------------------------------------------------
def generate_raw_data(n=500):
    os.makedirs(RAW_PATH, exist_ok=True)
    products = [
        ("P1", "Laptop", "Electronics", 55000),
        ("P2", "Phone", "Electronics", 20000),
        ("P3", "Headphones", "Accessories", 2500),
        ("P4", "Desk Chair", "Furniture", 7000),
        ("P5", "Notebook", "Stationery", 120),
    ]
    cities = ["Chennai", "Mumbai", "Delhi", "Pune", "Bangalore"]
    rows = []
    for i in range(1, n + 1):
        pid, pname, cat, price = random.choice(products)
        cid = random.randint(1, 60)
        rows.append({
            "order_id": f"O{i}",
            "customer_id": f"C{cid}",
            "customer_name": f"  Customer {cid} ",           # extra spaces
            "email": f"CUSTOMER{cid}@MAIL.COM",              # mixed case
            "city": random.choice(cities),
            "product_id": pid,
            "product_name": pname,
            "category": cat,
            "quantity": random.randint(1, 5),
            "unit_price": str(price),                        # string, not number
            "order_ts": (datetime(2026, 1, 1)
                         + timedelta(days=random.randint(0, 300),
                                     hours=random.randint(0, 23))).isoformat(),
        })
    rows += random.sample(rows, 30)                          # duplicates
    for r in random.sample(rows, 10):
        r["order_id"] = None                                 # bad records
    for r in random.sample(rows, 10):
        r["unit_price"] = "-1"                               # invalid price

    with open(f"{RAW_PATH}/orders.json", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------
# 2. BRONZE LAYER: raw data, stored as-is
# Why: keep an untouched copy plus audit columns so you can always
# reprocess Silver/Gold if your cleaning logic changes later.
# --------------------------------------------------------------------------
def build_bronze(spark):
    df = (
        spark.read.json(RAW_PATH)
        .withColumn("_ingest_ts", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_ingest_date", F.current_date())
    )
    (df.write.format("delta")
       .mode("append")                      # append only: never modify raw history
       .partitionBy("_ingest_date")         # partitioning = cheaper, faster reads
       .save(BRONZE))
    print(f"Bronze rows: {df.count()}")


# --------------------------------------------------------------------------
# 3. SILVER LAYER: cleaned, validated, deduplicated
# Why: analysts should never touch dirty data. Silver is the trusted layer.
# --------------------------------------------------------------------------
def build_silver(spark):
    bronze = spark.read.format("delta").load(BRONZE)

    cleaned = (
        bronze
        # fix types: price came in as a string
        .withColumn("unit_price", F.col("unit_price").cast("double"))
        .withColumn("quantity", F.col("quantity").cast("int"))
        .withColumn("order_ts", F.to_timestamp("order_ts"))
        .withColumn("order_date", F.to_date("order_ts"))
        # standardise text so joins and group-bys behave
        .withColumn("customer_name", F.trim("customer_name"))
        .withColumn("email", F.lower(F.trim("email")))
        # derived metric used later in Gold
        .withColumn("total_amount", F.col("quantity") * F.col("unit_price"))
        # data quality rules: drop rows that break business logic
        .filter(F.col("order_id").isNotNull())
        .filter(F.col("unit_price") > 0)
        .filter(F.col("quantity") > 0)
    )

    # Deduplicate: if the same order_id appears many times, keep the latest ingest.
    # Why a window function: dropDuplicates() picks a random row,
    # row_number() lets us choose which row wins.
    w = Window.partitionBy("order_id").orderBy(F.col("_ingest_ts").desc())
    silver = (
        cleaned.withColumn("rn", F.row_number().over(w))
        .filter("rn = 1")
        .drop("rn", "_source_file")
    )

    (silver.write.format("delta")
        .mode("overwrite")
        .partitionBy("order_date")          # queries usually filter by date
        .save(SILVER))
    print(f"Silver rows: {silver.count()}")


# --------------------------------------------------------------------------
# 4. GOLD LAYER: star schema for analytics
# Why: a star schema (one fact table + dimension tables) is what BI tools
# like Power BI / Tableau / Athena queries are fastest and simplest on.
# --------------------------------------------------------------------------
def build_gold(spark):
    silver = spark.read.format("delta").load(SILVER)

    # Dimension: customers (one row per customer)
    dim_customer = (
        silver.select("customer_id", "customer_name", "email", "city")
        .dropDuplicates(["customer_id"])
    )

    # Dimension: products
    dim_product = (
        silver.select("product_id", "product_name", "category")
        .dropDuplicates(["product_id"])
    )

    # Dimension: date (lets you slice by month, quarter, weekday)
    dim_date = (
        silver.select("order_date").distinct()
        .withColumn("date_key", F.date_format("order_date", "yyyyMMdd").cast("int"))
        .withColumn("year", F.year("order_date"))
        .withColumn("quarter", F.quarter("order_date"))
        .withColumn("month", F.month("order_date"))
        .withColumn("day_of_week", F.date_format("order_date", "EEEE"))
    )

    # Fact: one row per order, holds only keys and measures
    fact_sales = silver.select(
        "order_id", "customer_id", "product_id",
        F.date_format("order_date", "yyyyMMdd").cast("int").alias("date_key"),
        "quantity", "unit_price", "total_amount",
    )

    for name, df in [("dim_customer", dim_customer), ("dim_product", dim_product),
                     ("dim_date", dim_date), ("fact_sales", fact_sales)]:
        df.write.format("delta").mode("overwrite").save(f"{GOLD}/{name}")
        print(f"Gold {name}: {df.count()} rows")


# --------------------------------------------------------------------------
# 5. ANALYTICS: prove the Gold layer works
# --------------------------------------------------------------------------
def run_analytics(spark):
    for name in ["dim_customer", "dim_product", "dim_date", "fact_sales"]:
        spark.read.format("delta").load(f"{GOLD}/{name}").createOrReplaceTempView(name)

    print("\nRevenue by category and quarter:")
    spark.sql("""
        SELECT p.category, d.quarter, ROUND(SUM(f.total_amount), 2) AS revenue
        FROM fact_sales f
        JOIN dim_product p ON f.product_id = p.product_id
        JOIN dim_date d    ON f.date_key   = d.date_key
        GROUP BY p.category, d.quarter
        ORDER BY d.quarter, revenue DESC
    """).show()

    print("Top 5 cities by revenue:")
    spark.sql("""
        SELECT c.city, ROUND(SUM(f.total_amount), 2) AS revenue
        FROM fact_sales f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        GROUP BY c.city ORDER BY revenue DESC LIMIT 5
    """).show()


if __name__ == "__main__":
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    generate_raw_data()
    build_bronze(spark)
    build_silver(spark)
    build_gold(spark)
    run_analytics(spark)
    spark.stop()
