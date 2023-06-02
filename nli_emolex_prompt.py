import pandas as pd
import argparse
import statistics

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import classification_report

def compute_metrics(data, y_true, y_pred,output_file):
    print("Model perfomance\n")
    print(classification_report(y_true, y_pred))
    data['EmoLex'] = y_pred
    data.to_csv(output_file, sep="\t", index=False)

def compute_entailment(data, emolex_dic, prompts, output_file):
    
    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained('deberta-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('deberta-large-mnli')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text_list = data['text'].tolist()
    label_list = data['emotion'].tolist()
    id_text = data['id_text'].tolist()

    unique_labels = sorted(list(set(label_list)))
    print(unique_labels)

    df_prediction = pd.DataFrame()
    for i, text in enumerate(text_list):
        print("text number", i)
        premise = text
        probs_label = []
        dict_emo_conf = {x: {} for x in unique_labels}
        for label in unique_labels:
            probs_syn = []
            with torch.no_grad():
                for syn_hypo in emolex_dic[label]:
                    x = tokenizer.encode(premise, syn_hypo, return_tensors='pt',truncation_strategy='only_first')
                    x = x.to(device)
                    logits = model(x)[0]
                    entail_contradiction_logits = logits[:,[0,2]]
                    prob_label_is_true = entail_contradiction_logits.softmax(dim=1)[:,1]
                    probs_syn.append(prob_label_is_true.cpu().detach().numpy().tolist()[0])
                max_syn = statistics.mean(probs_syn)
            dict_emo_conf[label] = max_syn
        final_emo = max(dict_emo_conf, key=dict_emo_conf.get)      
        y_pred.append(final_emo)
        y_true.append(label_list[i])
        df_prediction.at[i, 'id_text'] = id_text[i]
        df_prediction.at[i, 'emotion'] = label_list[i]
        df_prediction.at[i, 'prediction'] = final_emo
    compute_metrics(data, y_true, y_pred, output_file)

def main():
    
    parser = argparse.ArgumentParser()
    
    # Requiered parameters
    parser.add_argument("--data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data file. Should contain the .tsv file for the emotion dataset.")
    
    parser.add_argument("--output_file",
                        default="ouput_dir",
                        type=str,
                        required=True,
                        help="The output file where the model predictions will be written in format .tsv.")            
  
    args = parser.parse_args()
    data_file= args.data_file
    output_file = args.output_file
    
    data = pd.read_csv(data_file, sep="\t")
    
    with open('./lexicon/emolex.pickle', 'rb') as handle:
        emolex_dic = pickle.load(handle)

    compute_entailment(data, emolex_dic, output_file)
    
if __name__ == "__main__":
    main()
