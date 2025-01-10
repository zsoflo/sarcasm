# Where to find codes
We provide in this GitHub repository all the codes used since the beginning of the project. Several of those codes were employed to implement methods that are directly mentionned in the 4th report of the project. Here is a list of all those codes and their location in the repository:
* All the data used for our project is gathered in the folder: https://github.com/zsoflo/sarcasm/tree/master/corpus
    - The source files that were used to extracted data from the iSarcasmEval dataset (Section 3.1.2 "Datasets and Data Gathering Overview") are located in: https://github.com/zsoflo/sarcasm/tree/master/corpus/isarcasm_dataset
    - The source files that were used to extracted data from the SIGN dataset (Section 3.1.2 "Datasets and Data Gathering Overview") are located in: https://github.com/zsoflo/sarcasm/tree/master/corpus/sign_dataset
    - The source files that were used to extracted data from the IronySarcasmAnalysisCorpus (Section 3.1.2 "Datasets and Data Gathering Overview") are located in: https://github.com/zsoflo/sarcasm/tree/master/corpus/irony_saracsm_dataset
    - The code that was used to extract the data from the iSarcasmEval and SIGN datasets (Section 3.1.2 "Datasets and Data Gathering Overview") is available here: https://github.com/zsoflo/sarcasm/blob/master/corpus/get_data.ipynb
    - The code that was used to extract the data from the IronySarcasmAnalysis Corpus (Section 3.1.2 "Datasets and Data Gathering Overview") is available here: https://github.com/zsoflo/sarcasm/blob/master/corpus/new_data.ipynb
    - The folder https://github.com/zsoflo/sarcasm/tree/master/corpus/merged_dataset compiles the main dataset files:
        - https://github.com/zsoflo/sarcasm/tree/master/corpus/merged_dataset/final_minidataset is the complete mini-Dataset (Section 3.1.2 "Datasets and Data Gathering Overview")
        - https://github.com/zsoflo/sarcasm/tree/master/corpus/merged_dataset/temporary_big_dataset is the big-Dataset without all the data coming from the IronySarcasmAnalysis Corpus (Section 3.1.2 "Datasets and Data Gathering Overview")
* The code used to run the rule-based classifier (Section 3.2.1 "Rule-based Classifier") is available here: https://github.com/zsoflo/sarcasm/blob/master/rule_based_approach/src/exp_final.ipynb
* The code used to run the rule-based interpreter (Section 3.2.2 "Rule-based Interpreter") is available here: https://github.com/zsoflo/sarcasm/blob/master/rule_based_approach/src/interpreter.ipynb
* The code used to fine-tune the stochastic baseline classification model (Section 3.3 "Baselines: Stochastic Models") is available here: https://github.com/zsoflo/sarcasm/blob/master/fine_tuning/src/classifier_finetuning.py
* The code used to fine-tune the stochastic baseline interpretation model (Section 3.3 "Baselines: Stochastic Models") is available here: https://github.com/zsoflo/sarcasm/blob/master/fine_tuning/src/interpreter_finetuning.py
* The guidelines applied for the annotation of the rule-based and stochastic models' interpretations (Section 3.4 "Evaluation Metrics") are available in the folder: https://github.com/zsoflo/sarcasm/tree/master/fluency_adequacy_annotation/guidelines
* The code used to run the two fine-tuned models is available here: https://github.com/zsoflo/sarcasm/blob/master/fine_tuning/src/add_predictions_to_csv.py
* The files containing the predictions of the rule-based and stochastic classifier (Section 4.1 "Classification Task") are available at https://github.com/zsoflo/sarcasm/blob/master/rule_based_approach/out/exp_final_classification_results_rule_based.csv and https://github.com/zsoflo/sarcasm/blob/master/fine_tuning/out/exp_final_classification_results_stochastic.csv respectively
* The files containing the predictions of the substitution, negation and stochastic interpreters (Section 4.2 "Interpretation Task") are available at https://github.com/zsoflo/sarcasm/blob/master/rule_based_approach/out/results_substitution_interpreter.csv, https://github.com/zsoflo/sarcasm/blob/master/rule_based_approach/out/results_substitution_interpreter.csv and https://github.com/zsoflo/sarcasm/blob/master/fine_tuning/out/results_stochastic_interpreter.csv, respectively
* The annotations themselves are available in the folder: https://github.com/zsoflo/sarcasm/tree/master/fluency_adequacy_annotation/annotations
* The code used to extract the sample of 1000 tweets (Section 7.2.1 "Sample Extraction") is available here : https://github.com/zsoflo/sarcasm/blob/master/corpus/get_data.ipynb
* The code used to analyze our annotations with respect to the original ones (Section 7.2.3 "Agreement with the Intended Sarcasm Annotations") is available here : https://github.com/zsoflo/sarcasm/blob/master/annotations/src/annotation_analysis_compare_inital.ipynb
* The code used to measure the IAA between us (Section 7.2.4 "Inter-Annotator Agreement (IAA)") is available here : https://github.com/zsoflo/sarcasm/blob/master/annotations/src/IAA_krippendorff.ipynb
* The analysis of the sentiment distances using each annotator's labels (Section 7.5 "Sentiment Distances Study") is available here: https://github.com/zsoflo/sarcasm/tree/master/sentiment_distances_analysis
<!--
# FOR US
The file we have to annotate is called : sample_to_annotate_without_duplicates\
really important :  WHEN YOU OPEN THE CSV FILE, IF A WINDOW APPEARS, ASKING YOU HOW YOU WANT YOUR DATA TO BE READ, PAY ATTENTION : IF YOU HAVE CHOICE ABOUT THE SEPARATOR(S) YOU CAN USE, USE THE TABS AND COMMAS BUT NOT THE SEMI-COLONS (NEVER). IF YOUR FILE IS STRANGE (SOME CELLS ARE SPLITTED INTO TWO DIFFERENT COLONS) GO TO THE SETTINGS AND CHANGE THE SEPARATORS AS MENTIONNED BEFORE. 
-->
