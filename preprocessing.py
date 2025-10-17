from pyspark.sql.functions import rand, col
from pyspark.ml.feature import StringIndexer, VectorAssembler

def clean_and_join(customers, orders, order_items, payments, products):
    # Drop nulls and duplicates
    orders = orders.dropna().dropDuplicates()
    order_items = order_items.dropna().dropDuplicates()
    products = products.dropna().dropDuplicates()
    payments = payments.dropna().dropDuplicates()
    customers = customers.dropna().dropDuplicates()

    # Join all tables
    order_details = order_items.join(products, "product_id", "left") \
        .join(orders.select("order_id", "customer_id", "order_status", "order_purchase_timestamp"), "order_id", "left") \
        .join(payments, "order_id", "left") \
        .join(customers, "customer_id", "left")

    # Simulate ~20% returns
    order_details = order_details.withColumn("is_returned", (rand() > 0.8).cast("double"))

    # Drop rows with nulls in critical columns
    order_details = order_details.dropna(subset=[
        "product_category_name", "price", "shipping_charges", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm", "is_returned"
    ])

    return order_details

def prepare_features(order_details):
    # Add product volume
    order_details = order_details.withColumn(
        "product_volume",
        col("product_length_cm") * col("product_width_cm") * col("product_height_cm")
    )

    # Index categorical columns
    category_indexer = StringIndexer(inputCol="product_category_name", outputCol="category_index", handleInvalid="keep")
    payment_indexer = StringIndexer(inputCol="payment_type", outputCol="payment_index", handleInvalid="keep")
    state_indexer = StringIndexer(inputCol="customer_state", outputCol="state_index", handleInvalid="keep")

    order_details = category_indexer.fit(order_details).transform(order_details)
    order_details = payment_indexer.fit(order_details).transform(order_details)
    order_details = state_indexer.fit(order_details).transform(order_details)

    # Assemble features
    assembler = VectorAssembler(
        inputCols=[
            "category_index", "price", "shipping_charges", "product_weight_g",
            "product_volume", "payment_index", "state_index"
        ],
        outputCol="features"
    )
    order_details = assembler.transform(order_details)

    order_details = order_details.select("features", "is_returned")
    return order_details