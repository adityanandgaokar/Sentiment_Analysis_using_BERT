import transformers
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import seaborn as sns 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from sklearn.model_selection import train_test_split
from torch import nn, optim
from collections import defaultdict
from torch.nn import functional as F
import logging
logging.basicConfig(level=logging.ERROR)
df = pd.read_csv("/content/Musical_instruments_reviews.csv")



def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  elif rating == 3:
    return 1
  else:
    return 2


df['sentiment'] = df.overall.apply(to_sentiment)
class_names = ['negative', 'neutral', 'positive']


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)


MAX_LEN = 160
BATCH_SIZE = 16

def GPReviewDataset(reviews, tokenizer, max_len):
  
  input_ids = []
  attention_mask = []

  for review in reviews:

    encoding = tokenizer.encode_plus(
      str(review),
      add_special_tokens=True,
      max_length=max_len,
      pad_to_max_length=True,
      return_attention_mask=True,
    )

    input_ids.append(encoding.get('input_ids'))
    attention_mask.append(encoding.get('attention_mask'))

  input_ids = torch.tensor(input_ids)
  attention_mask = torch.tensor(attention_mask)

  return input_ids, attention_mask


train_inputs, train_masks = GPReviewDataset(df.reviewText, tokenizer, MAX_LEN)


df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

train_labels = torch.tensor(df['sentiment'])

train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sample = RandomSampler(train_data)

training_data  = DataLoader(train_data, sampler=train_sample, batch_size=BATCH_SIZE)


bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

model = SentimentClassifier(len(class_names))
model = model.to(device)



EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(training_data) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()
  losses = []
  correct_predictions = 0
  for batch in data_loader:
    input_ids, attention_mask, targets = tuple(t.to(device) for t in batch)  
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(
    model,
    training_data,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  
 
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  
  if train_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = train_acc