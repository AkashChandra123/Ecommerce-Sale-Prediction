from data_loader import create_spark_session, load_data
from preprocessing import clean_and_join, prepare_features
from modeling import train_model, evaluate_model

spark = create_spark_session()
customers, orders, order_items, payments, products = load_data(spark)
order_details = clean_and_join(customers, orders, order_items, payments, products)
final_data = prepare_features(order_details)

model, train_data, test_data = train_model(final_data)
predictions = evaluate_model(model, test_data)