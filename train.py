from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from dataset_tokenizer import tokenized_datasets, checkpoint, data_collator, tokenizer

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

training_args = TrainingArguments(
  'test-trainer',
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=5,
  learning_rate=2e-5,
  weight_decay=0.01,
  use_mps_device=True
  )

trainer = Trainer(
  model,
  training_args,
  train_dataset=tokenized_datasets['train'],
  eval_dataset=tokenized_datasets['validation'],
  data_collator=data_collator,
  tokenizer=tokenizer
)

trainer.train()