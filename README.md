# Natural Language Inference Prompts for Zero-shot Emotion Classification in Text across Domains

This repository contains the code of the paper submitted to NAACL 2022 "Natural Language Inference Prompts for Zero-shot Emotion Classification in Text across Domains".

To download the emotion datasets and the lexicon resource, please go to the following links:

1. [ISEAR](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
2. [TEC](http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html)
3. [Blog:](http://saimacs.github.io/pubs/2007-TSD-paper.pdf) By contacting the authors
4. [Emolex](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)

To download the three pretrained entailment models (RoBERTa, BART, DeBERTa), please go to the following links:

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

**Important:** Before running the following experiments, when loading the datasets, rename the gold label with 'emotion' and the instance text with 'text'.

### Experiment 1

Get the predictions of the prompts `emo_name`, `expr_emo`, `feels_emo`, `wn_def`:

Run ```nli_prompts.py --data_file 'your_dataset_file.tsv' --transformer 'the_transformer_path' --output_file 'your_output_prediction_file.tsv' --prompt 'emo_name', 'expr_emo', 'feels_emo', 'wn_def'```

--transformer options: `roberta-large-mnli`, `facebook/bart-large-mnli`, `microsoft/deberta-v2-xlarge-mnli`

--prompt options: an unique prompt or a list of prompts: `emo_name`, `expr_emo`, `feels_emo`, `wn_def`

--output_file: The output directory where the model predictions will be written. Format required: '.tsv'

### Experiment 2

Get the predictions of the synonyms prompts `emo_s`, `expr_s`, `feels_s`:

Run ```nli_syn_prompts.py --data_file 'your_dataset_file.tsv' --output_file 'your_output_prediction_file.tsv' --prompt 'emo_s', 'expr_s', 'feels_s'```

--prompt options: an unique prompt or a list of prompts: `emo_s`, `expr_s`, `feels_s`

--output_file: The output directory where the model predictions will be written. Format required: '.tsv'

### Experiment 3

Get the predictions of the ensemble models:

**Important:** before running the following scripts make sure the predictions of the previous experiments (Experiment 1 using the DeBERTa model and Experiment 2) have been stored in your ouput files. Use these files in the following command:

```merge_predictions.py --exp1_predictions 'your_exp1_prediction_file.tsv' --exp2_predictions 'your_exp2_prediction_file.tsv' --output_file 'your_output_file.tsv'```

The predictions will be stored in 'your_output_file.tsv'. Use this file as an input of --data_file in the following command:

Run ```ensembles.py --data_file 'your_predictions_file.tsv' --output_file 'ensemble_predictions.tsv'```

--data_file: The input data file. Should contain the .tsv file with the predictions

--name_dataset options: tec, blog, isear

### Experiment 4

Get the predictions of the emolex prompt `emoLex`:

Run ```nli_emolex_prompt.py --data_file 'your_dataset_file.tsv' --output_file 'your_output_prediction_file.tsv'```

--data_file: The emotion dataset path file

--output_file: The output directory where the model predictions will be written. Format required: '.tsv'
