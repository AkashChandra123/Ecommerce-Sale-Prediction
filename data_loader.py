import os
from pyspark.sql import SparkSession

def create_spark_session():
    return SparkSession.builder.appName("Ecommerce Analysis").getOrCreate()

def load_data(spark):
    # Get absolute path to the data folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data")

    orders = spark.read.csv(os.path.join(data_path, "df_Orders.csv"), header=True, inferSchema=True)
    order_items = spark.read.csv(os.path.join(data_path, "df_OrderItems.csv"), header=True, inferSchema=True)
    customers = spark.read.csv(os.path.join(data_path, "df_Customers.csv"), header=True, inferSchema=True)
    payments = spark.read.csv(os.path.join(data_path, "df_Payments.csv"), header=True, inferSchema=True)
    products = spark.read.csv(os.path.join(data_path, "df_Products.csv"), header=True, inferSchema=True)

    return customers, orders, order_items, payments, products