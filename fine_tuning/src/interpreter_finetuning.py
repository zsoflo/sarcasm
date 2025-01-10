from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
from torch.cuda import empty_cache
from torch.cuda import is_available
from torch.utils.data import Dataset
from torch import tensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate

# Importing the data 
df = pd.read_csv("shuffled.csv")
df = df[~ df['tweet'].isna()]
df = df[~ df['rephrase'].isna() ]
df["repeat"] = df.apply( lambda x : int(x["rephrase"].strip() == x["tweet"].strip()) , axis = 1 )
df = df[df["repeat"] == 0]
df = df.drop(["repeat"], axis = 1)

# pack the rephrase per tweet
uniq_tweets = list(dict.fromkeys(df['tweet'].tolist()))
tweets_interpret = {"tweet" : uniq_tweets}
tweets_interpret["rephrase"] = [ (df[ df['tweet'] == tw ])["rephrase"].tolist()  for tw in uniq_tweets ]
df_packed = pd.DataFrame(tweets_interpret)

# Selecting train test 
in_test_tweets = pd.read_csv("tmp_exp4_test_final_27.csv")['tweet'].tolist()
test_size = 0.2
mandatory_in_test = [x for x in uniq_tweets if x in in_test_tweets]
other_tweets = [x for x in uniq_tweets if x not in in_test_tweets]
constraint_ratio = len(mandatory_in_test) / len(uniq_tweets)
if constraint_ratio > test_size :
    raise ValueError(f"Impossible : contsraint ratio = {round(constraint_ratio, 2)}")
test_size = test_size - constraint_ratio
train_list, test_list =  train_test_split(other_tweets, test_size=test_size, random_state=42)
test_list = test_list + mandatory_in_test
if len(test_list) + len(train_list) != len(uniq_tweets) :
    raise ValueError(f"Problem : train + test = {len(test_list) + len(train_list)} != uniq = { len(uniq_tweets)}")
df_train = df_packed[ df_packed['tweet'].isin( train_list )  ]
df_test = df_packed[ df_packed['tweet'].isin( test_list ) ]

model_name = "Glowcodes/autotrain-code-switched-tweets-sum-42325108532"
max_len = 500

if __name__ == "__main__" :

    empty_cache()

    # put the dataframe into the orginal format (one rephrase per row)
    df_train = df[ df['tweet'].isin( df_train['tweet'].tolist() ) ]
    df_test = df[ df['tweet'].isin( df_test['tweet'].tolist() ) ]

    # Define the device
    device = "cuda:0" if is_available() else "cpu"

    # Create info file
    with open("info_training_mt.txt", 'w') as f :
    	ratio_train = df_train["sarcastic"].mean()
    	ratio_test = df_test["sarcastic"].mean()
    	f.write(f"Train : {round(ratio_train, 4) * 100}%\n")
    	f.write(f"Test : {round(ratio_test, 4) * 100}%\n")
    	f.write(f"Deivice : {device}\n")

    # Pick the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                    output_hidden_states=True,
                                                    output_attentions=True)
    model = model.to(device)

    # Load the tokenizer and tokenize

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(df_train["tweet"].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
    test_encodings = tokenizer(df_test["tweet"].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)

    train_labels = tokenizer(df_train["rephrase"].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
    test_labels = tokenizer(df_test["rephrase"].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)

    # Formatting the data (Dataset format)
    class TweetMTDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels["input_ids"]

        def __getitem__(self, idx):
            item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TweetMTDataset(train_encodings, train_labels )
    test_dataset = TweetMTDataset(test_encodings, test_labels )

    # Define metrics computation function
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = blue.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        result["rouge"] = result["score"]
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result


    # Training
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=2,              # total number of training epochs
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
        # compute_metrics=compute_metrics  # The function that will be used to compute metrics at evaluation
    )
    

    train_output = trainer.train()

    # Complete info file
    with open("info_training_mt.txt", 'a') as f :
    	f.write("---------\nTraining\n---------\n")
    	f.write(f"output:{str(train_output)}")

    # Saving the model
    trainer.save_model("model_mt_save_file")