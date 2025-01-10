from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, BartForConditionalGeneration, pipeline
from interpreter_finetuning import model_name, max_len
from torch.cuda import is_available
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import evaluate
import math

tqdm.pandas()

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

device = "cuda:0" if is_available() else "cpu"

clf_model_name = "gaunernst/bert-tiny-uncased"
mt_model_name = "Glowcodes/autotrain-code-switched-tweets-sum-42325108532"

clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_name)
mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_name)

clf_model = AutoModelForSequenceClassification.from_pretrained("model_clf_save_file")
clf_model = clf_model.to(device)

mt_model = BartForConditionalGeneration.from_pretrained("model_mt_save_file")
mt_model = mt_model.to(device)

def predict_clf(input_text) :
	input_encodings = clf_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
	output = clf_model(**input_encodings )
	res = int(list(output.logits.argmax(-1))[0])
	return res

def predict_mt(input_text) :
	input_encodings = mt_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
	# best : repetition_penalty=2.0, top_p=0.8, top_k=0.8, temperature=0.8
	# best : repetition_penalty=2.0, top_p=0.8, top_k=500, temperature=0.8, do_sample=True
	# best :  repetition_penalty=2.0, top_p=0.5, top_k=200, temperature=0.5, do_sample=True
	prediction = mt_model.generate(input_encodings["input_ids"], repetition_penalty=2.0, top_p=0.8, top_k=500, temperature=0.8)
	return mt_tokenizer.batch_decode(prediction[0], skip_special_tokens=True)[0]

if len(sys.argv) >= 2 :
	csvfilename = sys.argv[1]
else :
	csvfilename = input("Write csv file name : ")

if csvfilename[-4:] != ".csv" :
	csvfilename += ".csv"

df = pd.read_csv(csvfilename)

print("Classifier")
df["stochastic_prediction_label"] = df.progress_apply(lambda x : predict_clf(x["modified_tweet"]), axis=1)

print("Interpreter")
df["stochastic_prediction_rephrase"] = df.progress_apply(lambda x : predict_mt(x["modified_tweet"]) if x["stochastic_prediction_label"] == 1 else None, axis=1)

df.to_csv(csvfilename[:-4] + "_with_predictions.csv", index=False)