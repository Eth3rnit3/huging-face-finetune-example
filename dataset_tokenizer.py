from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets  = load_dataset('glue', 'mrpc')
checkpoint    = 'bert-base-cased'
tokenizer     = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(example):
  return tokenizer(
    example['sentence1'], example['sentence2'], padding="max_length", truncation=True, max_length=128
  )
  

tokenized_datasets = raw_datasets.map(tokenize, batched=True)
# tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence1', 'sentence2'])
# tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
# tokenized_datasets = tokenized_datasets.with_format('torch')

data_collator = DataCollatorWithPadding(tokenizer)