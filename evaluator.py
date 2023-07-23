import numpy as np
from datasets import load_metric
from dataset_tokenizer import tokenized_datasets
from train import trainer

# Evaluation
predictions = trainer.predict(tokenized_datasets['validation'])
print(predictions.predictions.shape, predictions.label_ids.shape)

# Metrics
metric = load_metric('glue', 'mrpc')
preds = np.argmaxx(predictions.predictions, axis=-1)
print(metric.compute(predictions=preds, references=predictions.label_ids))