import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 데이터셋 로드
data = pd.read_table('D:/Topic_Modeling/ratings_train.txt')
data = data.iloc[:,1:]
data = data.dropna(how = 'any')

train_texts, val_texts, train_labels, val_labels = train_test_split(data['document'].tolist(), data['label'].tolist(), test_size=0.2)

# Hugging Face Dataset으로 변환
train_dataset = Dataset.from_dict({'document': train_texts, 'label': train_labels})
val_dataset = Dataset.from_dict({'document': val_texts, 'label': val_labels})

# DistilBERT 모델과 토크나이저 로드
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 데이터셋 토크나이즈
def tokenize_function(examples):
    return tokenizer(examples['document'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Trainer 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 모델 학습
trainer.train()

# 모델 저장
model.save_pretrained('./distilbert_finetuned')
tokenizer.save_pretrained('./distilbert_finetuned')
