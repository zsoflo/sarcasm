import pandas as pd
import os

filepaths = [f for f in os.listdir(".") if f.endswith('.csv')]
df = pd.DataFrame() # empty
# Gather everything into a single dataframe
for file in filepaths:
    if df.empty:
        df = pd.read_csv(file)
        df = df[~df['annotation'].isnull()] # remove null annotations
        df = df.rename(columns={'annotation': file[:-4]})
        df = df.drop(columns=['comments'], errors='ignore')
    else:
        new_csv = pd.read_csv(file)[['ID', 'annotation']]
        new_csv = new_csv[~new_csv['annotation'].isnull()] # remove null annotations
        new_csv = new_csv.rename(columns={'annotation': file[:-4]})
        df = pd.merge(df, new_csv, how='inner', on='ID')

print(df)

print()
print("How many tweets were marked as sarcastic by this annotator?")
for column in df.columns[2:]:
    sarcastic = df[column].sum()
    print(f'  {column}: {sarcastic} ({sarcastic / df.shape[0] * 100.0 :.2f} %)')

agreement = df[df.nunique(1).eq(3)] # 3 because the 3 different values on each line should be : id, tweet, and only one value that's the same for each annotator

agreement_sarc = agreement.loc[agreement[filepaths[0][:-4]] == 1.0]
agreement_not_sarc = agreement.loc[agreement[filepaths[0][:-4]] == 0.0]

print()
print(f"All annotators agreed that these {agreement_sarc.shape[0]} tweets are sarcastic:")
print(', '.join(map(str, agreement_sarc['ID'])))

print()
print(f"All annotators agreed that these {agreement_not_sarc.shape[0]} tweets are not sarcastic:")
print(', '.join(map(str, agreement_not_sarc['ID'])))

