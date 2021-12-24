import argparse
import statistics
import pandas as pd
import numpy as np

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import classification_report

def compute_metrics(data, y_true, y_pred, probs_emotions, id_prompt, output_file):
    print(classification_report(y_true, y_pred))
    data[id_prompt] = y_pred
    data['prob_'+id_prompt] = probs_emotions
    data.to_csv(output_file, sep="\t", index=False)

def compute_entailment(data, template_emo_name, template_expr_emo, template_feels_emo, prompts, output_file):
    
    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v2-xlarge-mnli')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v2-xlarge-mnli')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print("Predicting data using", prompts, "...")

    text_list = data['text'].tolist()
    label_list = data['emotion'].tolist()
    unique_labels = sorted(list(set(label_list)))
    
    for id_prompt in prompts:
        probs_emotions = []
        y_true = []
        y_pred = []
        for i, text in enumerate(text_list):
            premise = text
            dict_emo_conf = {x: {} for x in unique_labels}
            for label in unique_labels:
                probs_syn = []
                with torch.no_grad():
                    if id_prompt == "emo_s":
                        template = template_emo_name
                        context = ''
                    if id_prompt == "expr_emo":
                        template = template_expr_emo
                        context = 'This text expresses '
                    elif id_prompt == "feels_emo":
                        template = template_feels_emo
                        context = 'This person feels '
                    for syn_prompt in template[label]:
                        x = tokenizer.encode(premise, context + syn_prompt, return_tensors='pt',truncation_strategy='only_first')
                        x = x.to(device)
                        logits = model(x)[0]
                        entail_contradiction_logits = logits[:,[0,2]]
                        prob_label_is_true = entail_contradiction_logits.softmax(dim=1)[:,1]
                        probs_syn.append(prob_label_is_true.cpu().detach().numpy().tolist()[0])
                    mean = statistics.mean(probs_syn)
                dict_emo_conf[label] = mean
            probs_emotions.append(list(dict_emo_conf.values()))
            final_emo = max(dict_emo_conf, key=dict_emo_conf.get)           
            y_pred.append(final_emo)
            y_true.append(label_list[i])
        print("Model performance:")                                                 
        compute_metrics(data, y_true, y_pred, probs_emotions, id_prompt, output_file)

def main():
    
    parser = argparse.ArgumentParser()
    
    # Requiered parameters
    parser.add_argument("--data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data file. Should contain the .tsv file for the emotion dataset.")
    
    parser.add_argument("--output_file",
                        default="ouput_file",
                        type=str,
                        required=True,
                        help="The output file where the model predictions will be written.")
    
    parser.add_argument("--prompt", 
                        default=["EmoName"], 
                        nargs="+",
                        required=True,
                        help="The prompt or list of prompts to interpret the emotion selected in the list: \
                        emo_s, expr_s, feelss")
    
    args = parser.parse_args()
    data_file = args.data_file
    prompts = args.prompt
    output_file = args.output_file
    
    template_emo_s = {
        'sadness': ['sadness', 'unhappy', 'grief', 'sorrow', 'loneliness', 'depression'],
        'joy': ['joy', 'achievement', 'pleasure', 'awesome', 'happy', 'blessed'],
        'anger': ['anger', 'annoyance', 'rage', 'outrage', 'fury', 'irritation'],
        'disgust': ['disgust', 'loathing', 'bitter', 'ugly', 'repugnance', 'revulsion'],
        'fear': ['fear', 'horror', 'anxiety', 'terror', 'dread', 'scare'],
        'surprise': ['surprise', 'astonishment', 'amazement', 'impression', 'perplexity', 'shock'],
        'shame': ['shame', 'humiliate', 'embarrassment', 'disgrace', 'dishonor', 'discredit'],
        'guilt': ['guilt', 'culpability', 'blameworthy', ' responsibility', 'misconduct', 'regret'],
        'noemo': ['others', 'no emotion']
    }

    template_expr_s = {
        'sadness': ['sadness', 'unhappiness', 'grief', 'sorrow', 'loneliness', 'depression'],
        'joy': ['joy', 'an achievement', 'pleasure', 'the awesome', 'happiness', 'the blessing'],
        'anger': ['anger', 'annoyance', 'rage', 'outrage', 'fury', 'irritation'],
        'disgust': ['disgust', 'loathing', 'bitterness', 'ugliness', 'repugnance', 'revulsion'],
        'fear': ['fear', 'horror', 'anxiety', 'terror', 'dread', 'scare'],
        'surprise': ['surprise', 'astonishment', 'amazement', 'impression', 'perplexity', 'shock'],
        'shame': ['shame', 'humiliation', 'embarrassment', 'disgrace', 'dishonor', 'discredit'],
        'guilt': ['guilt', 'culpability', 'responsibility', 'blameworthy', 'misconduct', 'regret'],
        'noemo': ['others','no emotion']
    }
    template_feels_s = {
        'sadness': ['sadness', 'unhappy', 'grieved', 'sorrow', 'lonely', 'depression'],
        'joy': ['joyful', 'accomplished',  'pleasure', 'awesome', 'happy', 'blessed'],
        'anger': ['anger', 'annoyed', 'rage', 'outraged', 'furious', 'irritated'],
        'disgust': ['disgusted', 'loathing', 'bitter', 'ugly', 'repugnance', 'revulsion'],
        'fear': ['fear', 'horror', 'anxiety', 'terrified', 'dread', 'scared'],
        'surprise': ['surprised', 'astonishment', 'amazement', 'impressed', 'perplexed', 'shocked'],
        'shame': ['shameful', 'humiliated', 'embarrassed', 'disgraced', 'dishonored', 'discredit'],
        'guilt': ['guilty', 'culpable', 'responsible', 'blame', 'misconduct', 'regretful'],
        'noemo': ['others', 'no emotion']
    }
    
    data = pd.read_csv(data_file, sep="\t")
    
    compute_entailment(data, template_emo_s, template_expr_s, template_feels_s, prompts, output_file)
    
if __name__ == "__main__":
    main()
