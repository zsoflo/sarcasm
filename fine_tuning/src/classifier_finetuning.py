from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
from torch.cuda import empty_cache
from torch.cuda import is_available
from torch.utils.data import Dataset
from torch import tensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

empty_cache()

max_len=500

# Importing the data (with contraints on test)
in_test_tweets = pd.read_csv("tmp_exp4_test_final_27.csv")['tweet'].tolist()
test_size = 0.2
df = pd.read_csv("shuffled.csv")[["tweet", "sarcastic"]]
df2 = pd.read_csv("exp_final.csv")[["modified_tweet", "sarcastic"]]
df2 = df2.rename(columns={"modified_tweet" : "tweet"})
df = pd.concat([df, df2], axis=0)

df = df[~ df['tweet'].isna()]
df.drop_duplicates(inplace=True, subset=["tweet"])
uniq_tweets = list(dict.fromkeys(df['tweet'].tolist()))
mandatory_in_test = [x for x in uniq_tweets if x in in_test_tweets]
other_tweets = [x for x in uniq_tweets if x not in in_test_tweets]
constraint_ratio = len(mandatory_in_test) / len(uniq_tweets)
if constraint_ratio > test_size :
    print(f"Test constraint ratio = {round(constraint_ratio, 3)} > {round(test_size, 3)}")
    train_list = other_tweets
    test_list = mandatory_in_test
else :
    test_size = test_size - constraint_ratio
    train_list, test_list =  train_test_split(other_tweets, test_size=test_size, random_state=42)
    test_list = test_list + mandatory_in_test
if len(test_list) + len(train_list) != len(uniq_tweets) :
    raise ValueError(f"Problem : train + test = {len(test_list) + len(train_list)} != uniq = { len(uniq_tweets)}")

df_train = df[ df['tweet'].isin( train_list )  ]
df_test = df[ df['tweet'].isin( test_list ) ]

# Define the device
device = "cuda:0" if is_available() else "cpu"

# Create info file
with open("info_training.txt", 'w') as f :
	ratio_train = df_train["sarcastic"].mean()
	ratio_test = df_test["sarcastic"].mean()
	f.write(f"Train : {round(ratio_train, 4) * 100}%\n")
	f.write(f"Test : {round(ratio_test, 4) * 100}%\n")
	f.write(f"Deivice : {device}\n")

# Pick the model
model_name = "gaunernst/bert-tiny-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=2,
                                                           output_hidden_states=True,
                                                           output_attentions=True
)
model = model.to(device)

# Load the tokenizer and tokenize

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(df_train["tweet"].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
test_encodings = tokenizer(df_test["tweet"].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

# Formatting the data (Dataset format)
class TweetClassDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
train_dataset = TweetClassDataset(train_encodings, df_train["sarcastic"].tolist() )
test_dataset = TweetClassDataset(test_encodings, df_test["sarcastic"].tolist())

# Define metrics computation function
def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions[0]
    # print( f"{len(labels)} = {len(predictions)}" )
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    save_total_limit=1,              # limit the total amount of checkpoints. Deletes the older checkpoints.
    dataloader_pin_memory=False,  # Whether you want to pin memory in data loaders or not. Will default to True
    # evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
    evaluation_strategy="steps",
    logging_steps=50,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    compute_metrics=compute_metrics  # The function that will be used to compute metrics at evaluation
)

train_output = trainer.train()

# Complete info file
with open("info_training.txt", 'a') as f :
	f.write("---------\nTraining\n---------\n")
	f.write(f"output:{str(train_output)}")

# Saving the model
trainer.save_model("model_clf_save_file")

def predict_clf(input_text) :
    input_encodings = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
    output = model(**input_encodings )
    res = int(list(output.logits.argmax(-1))[0])
    return res

df_test["predicted_label"] = df_test.apply( lambda x : predict_clf(x["tweet"]) , axis=1 )
df_test.to_csv("classification_test_dataset_output.csv")
