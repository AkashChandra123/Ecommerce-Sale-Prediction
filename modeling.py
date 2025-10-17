from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_model(final_data):
    train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
    model = LogisticRegression(labelCol="is_returned", featuresCol="features")
    model = model.fit(train_data)
    return model, train_data, test_data

def evaluate_model(model, test_data):
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="is_returned")
    accuracy = evaluator.evaluate(predictions)
    return predictions, accuracy