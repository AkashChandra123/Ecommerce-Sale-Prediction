from data_loader import create_spark_session, load_data
from preprocessing import clean_and_join, prepare_features

spark = create_spark_session()
customers, orders, order_items, payments, products = load_data(spark)
order_details = clean_and_join(customers, orders, order_items, payments, products)
final_data = prepare_features(order_details)

final_data.show(5)