import argparse
import statistics
import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import classification_report

def compute_metrics(data, y_true, y_pred, probs_emotions, id_prompt, output_file):
    print(classification_report(y_true, y_pred))
    data[id_prompt] = y_pred
    data['prob_'+id_prompt] = probs_emotions
    data.to_csv(output_file, sep="\t", index=False)

def compute_entailment(data, transfomer, template, prompts, output_file):
    
    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(transfomer)
    tokenizer = AutoTokenizer.from_pretrained(transfomer)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print("Predicting data using", prompts, "...")

    text_list = data['text'].tolist()
    label_list = data['emotion'].tolist()
    unique_labels = sorted(list(set(label_list)))
    
    for id_prompt in prompts:
        y_pred = []
        y_true = []
        probs_emotions = []
        for index, text in enumerate(text_list):
            premise = text
            probs = []
            with torch.no_grad():
                for label in unique_labels:
                    if id_prompt == "emo_name":
                        # to convert the noemmo label of Blog into the natural language name "no emotion"
                        if label == "noemo":
                            label = "no emotion"
                        prompt = label
                    if id_prompt == "expr_emo":
                        prompt = template[label][0]
                    elif id_prompt == "feels_emo":
                        prompt = template[label][1]
                    elif id_prompt == "wn_def":
                        prompt = template[label][2]
                    x = tokenizer.encode(premise, prompt, return_tensors='pt',truncation_strategy='only_first')
                    x = x.to(device)
                    logits = model(x)[0]
                    entail_contradiction_logits = logits[:,[0,2]]
                    prob_label_is_true = entail_contradiction_logits.softmax(dim=1)[:,1]
                    probs.append(prob_label_is_true.detach().cpu().numpy()[0])
            y_pred.append(unique_labels[np.argmax(np.array(probs))])
            probs_emotions.append(probs)
            y_true.append(label_list[index])
        print("\nModel performance:")
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
                        emo_name, expr_emo, feels_emo, wn_def, emo_s, expr_s, feels_s")
    
    parser.add_argument("--transformer", 
                        default="roberta-large-mnli",
                        required=True,
                        help="Transformer pre-trained model selected in the list: roberta-large-mnli, \
                        microsoft/deberta-v2-xlarge-mnli, facebook/bart-large-mnli")              
  
    args = parser.parse_args()
    data_file = args.data_file
    output_file = args.output_file
    transfomer = args.transformer
    prompts = args.prompt
    
    template = {
                'sadness': ['This text expresses sadness', 
                            'This person feels sad', 
                            'This pserson expresses emotions experienced when not in a state of well-being'],
                'joy':     ['This text expresses joy' ,
                            'This person feels joyful', 
                            'This person expresses a feeling of great pleasure and happiness.'],
                'anger':   ['This text expresses anger', 
                            'This person feels angry', 
                            'This person expresses a strong feeling of annoyance, displeasure, or hostility'],
                'disgust': ['This text expresses disgust', 
                            'This person feels disgusted', 
                            'This person expresses a feeling of revulsion or strong disapproval aroused by something unpleasant or offensive'],
                'fear':    ['This text expresses fear', 
                             'This person is afraid of something', 
                             'This person expresses an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat'],
                'surprise': ['This text expresses surprise', 
                             'This person feels surprised', 
                             'This person expresses a feeling of mild astonishment or shock caused by something unexpected'],
                'shame':    ['This text expresses shame', 
                             'This person feels shameful', 
                             'This person expresses a painful feeling of humiliation or distress caused by the consciousness of wrong or foolish behavior'],
                'guilt':    ['This text expresses guilt', 
                             'This person feels guilty', 
                             'This person expresses a feeling of having done wrong or failed in an obligation'],
                'noemo':    ['This text does not expresses any emotion', 
                             'This person does not feel any emotion', 
                             'This person does not feel any emotion']
                }
    
    data = pd.read_csv(data_file, sep="\t")
    
    compute_entailment(data, transfomer, template, prompts, output_file)
    
if __name__ == "__main__":
    main()

#python roberta_prediction.py --data_file '/mnt/beegfs/fmplaza/Datasets/Emotions/unify-emotion-datasets/datasets_tsv/TEC' --transformer 'roberta-large-mnli' --output_file 'predictions_main_prompts.tsv' --id_prompt 'EmoName'
