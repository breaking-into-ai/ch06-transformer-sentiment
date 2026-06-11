"""Chapter 6 - fine-tune DistilBERT on IMDB sentiment. GPU strongly recommended."""
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
args = TrainingArguments(
    output_dir="./cinematch-sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # low LR for fine-tuning - don't disrupt pre-training
    weight_decay=0.01,
    load_best_model_at_end=True,
)
trainer = Trainer(model=model, args=args,
                  train_dataset=tokenized["train"], eval_dataset=tokenized["test"])
trainer.train()  # typically 93%+ accuracy in under 30 minutes on a single GPU

# Use it:
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="./cinematch-sentiment")
for review in ["This movie was an absolute masterpiece.",
               "Worst two hours of my life."]:
    result = classifier(review)[0]
    print(f"{review[:50]} -> {result['label']} ({result['score']:.1%})")
