# Natural Language Inference Prompts for Zero-shot Emotion Classification in Text across Domains

This repository contains the code of the paper submitted to NAACL 2022 "Natural Language Inference Prompts for Zero-shot Emotion Classification in Text across Domains".

To download the emotion datasets, please go to these URLs:

1. [ISEAR](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
2. [TEC](http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html)
3. [Blog:](http://saimacs.github.io/pubs/2007-TSD-paper.pdf) By contacting the authors.

To download the three pretrained entailment models (RoBERTa, BART, DeBERTa), please go to these URLs:

1. [RoBERTa](https://huggingface.co/roberta-large-mnli)
2. [BART](https://huggingface.co/facebook/bart-large-mnli)
3. [DeBERTa](https://huggingface.co/microsoft/deberta-v2-xlarge-mnli)

## Instructions to run the code

### Requirements

* python 3.8.10
* transformers 4.6.1
* torch 1.7.1
* scikit-learn 0.24.2
* pandas 1.0.3
* numpy 1.18.2

### Experiment 1

Get the predictions of the prompts `emo_name`, `expr_emo`, `feels_emo`, `wn_def`:

Run ```nli_main_prompts.py --data_dir 'your_dataset_file' --transformer 'the_transformer_path' --output_dir 'your_output_prediction_file' --id_prompt 'emo_name', 'expr_emo', 'feels_emo', 'wn_def'```

--transformer options: `roberta-large-mnli`, `facebook/bart-large-mnli`, `microsoft/deberta-v2-xlarge-mnli`

--id_prompt options: an unique prompt or a list of prompts: `emo_name`, `expr_emo`, `feels_emo`, `wn_def`

--output_file: The output directory where the model predictions will be written

### Experiment 2

Get the predictions of the synonyms prompts `emo_s`, `expr_s`, `feels_s`:

Run ```nli_syn_prompts.py --data_dir 'your_dataset_file' --output_dir 'your_output_prediction_file' --id_prompt 'emo_s', 'expr_s', 'feels_s'```

--id_prompt options: an unique prompt or a list of prompts: `emo_s`, `expr_s`, `feels_s`
--output_file: The output directory where the model predictions will be written

### Experiment 3

Get the predictions of the ensemble models:

Important: before running the following scripts make sure the predictions of the previous experiments (Experiment 1 and Experiment 2) have been stored in your ouput files. Use these files to run the following command:

```merge_predictions.py --predictions_exp1 'your_exp1_prediction_file' --predictions exp2 'your_exp2_prediction_file'' --output_file 'your_output_file'```

The predictions will be stored in 'your_output_file'. Use this file in the following command:

Run ```esembles.py --data_dir 'your_dataset_file' --output_dir 'your_output_prediction_file'

--data_dir: The input data dir. Should contain the .tsv file with the predictions.
--name_dataset options: tec, blog, isear
--output_file: The output directory where the model predictions will be written

### Experiment 4

Get the predictions of the emolex prompt `emoLex`:

After merging the predictions of Experiment 1 and Experiment 2, please run the following code:

Run ```nli_emolex_prompt.py --data_dir 'your_dataset_file' --output_dir 'your_output_prediction_file'

--id_prompt options: an unique prompt or a list of prompts: `emo_s`, `expr_s`, `feels_s`

--output_file: The output directory where the model predictions will be written
