import pandas as pd
import argparse
from collections import defaultdict
import numpy as np
from ast import literal_eval

from sklearn.metrics import classification_report

def compute_metrics(data, y_true, y_pred, output_file, t_ensemble):
    data[t_ensemble] = y_pred
    data.to_csv(output_file, sep="\t", index=False)
    print(classification_report(y_true, y_pred))

def ensemble_strategy(data, name_dataset, output_file):
    index2emo_blog = {0: 'anger', 1: 'disgust', 2:'fear', 3:'joy', 4:'noemo', 5:'sadness', 6:'surprise'}
    index2emo_tec = {0: 'anger', 1: 'disgust', 2:'fear', 3:'joy', 4:'sadness', 5:'surprise'}
    index2emo_isear = {0: 'anger', 1: 'disgust', 2:'fear', 3:'guilt', 4:'joy', 5:'sadness', 6:'shame'}
    corpus_name_dic = {'tec': index2emo_tec, 'blog': index2emo_blog, 'isear':index2emo_isear}
    y_pred = []
    y_true=[]
    for index, row in data.iterrows():
        dict = defaultdict(float)
        list_pred = [row['emo_name'], row['emo_s'], row['expr_emo'], row['expr_s'], row['feels_emo'], row['feels_s'], row['wn_def']]
        list_prob = [row['prob_emo_name'], row['prob_emo_s'], row['prob_expr_emo'], row['prob_expr_s'], row['prob_feels_emo'], row['prob_feels_s'], row['prob_wn_def']]
        number_models = len(list_prob)
        list_prob = np.array(list_prob)
        average_probs = np.sum(list_prob, axis=0)/number_models
        maxElement = np.amax(average_probs)
        maxElement_pos = np.where(average_probs == np.amax(average_probs))
        maxElement_pos = maxElement_pos[0][0]
        final_emo = corpus_name_dic[name_dataset][maxElement_pos]
        y_pred.append(final_emo)
        y_true.append(row['emotion'])
        t_ensemble = 'ensemble'
    print("\nEnsemble performance\n")
    compute_metrics(data, y_true, y_pred, output_file, t_ensemble)
    
def oracle(data, name_dataset, output_file):
    TP = 0
    FN = 0
    y_true = data.emotion
    y_pred = []
    for index, row in data.iterrows():
        label_gold = row['emotion']
        if (label_gold == row['emo_name'] or label_gold == row['expr_emo'] or label_gold == row['feels_emo'] or label_gold == row['wn_def'] or label_gold == row['emo_s']or label_gold == row['expr_s'] or label_gold == row['feels_s']):
            TP += 1
            y_pred.append(label_gold)
        else:
            FN += 1
            y_pred.append(row['expr_s'])
    t_ensemble = 'oracle'
    
    print("Oracle ensemble performance\n")
    compute_metrics(data, y_true, y_pred, output_file, t_ensemble)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file",
                        type=str,
                        required=True,
                        help="The input data file. Should contain the .tsv file with the predictions.")

    parser.add_argument("--name_dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the emotion dataset. List of datasets: blog, isear, tec")

    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model predictions will be written.")

    args = parser.parse_args()

    name_dataset = args.name_dataset
    output_file = args.output_file

    data = pd.read_csv(args.data_file, sep="\t")

    data['prob_emo_name'] = data['prob_emo_name'].apply(literal_eval)
    data['prob_expr_emo'] = data['prob_expr_emo'].apply(literal_eval)
    data['prob_feels_emo'] = data['prob_feels_emo'].apply(literal_eval)
    data['prob_wn_def'] = data['prob_wn_def'].apply(literal_eval)
    data['prob_emo_s'] = data['prob_emo_s'].apply(literal_eval)
    data['prob_expr_s'] = data['prob_expr_s'].apply(literal_eval)
    data['prob_feels_s'] = data['prob_feels_s'].apply(literal_eval)

    ensemble_strategy(data, name_dataset, output_file)
    oracle(data, name_dataset, output_file)

if __name__ == "__main__":
    main()
