import argparse
import pandas as pd

parser = argparse.ArgumentParser()

# Requiered parameters
parser.add_argument("--exp1_predictions",
                    default=None,
                    type=str,
                    required=True,
                    help="The file with the predictions of the Experiment 1 using the DeBERTa model.")


parser.add_argument("--exp2_predictions",
                    default=None,
                    type=str,
                    required=True,
                    help="The file with the predictions of the Experiment 2 using the DeBERTa model.")

parser.add_argument("--output_file",
                    default="ouput_file",
                    type=str,
                    required=True,
                    help="The output file with the predictions of the Exp 1 and Exp2.")
  
args = parser.parse_args()
    
exp1_predictions = pd.read_csv(args.exp1_predictions, sep=" ")
exp2_predictions = pd.read_csv(args.exp2_predictions, sep=" ")

exp2_predictions = exp2_predictions[['emo_s', 'expr_s', 'feels_s', 'prob_emo_s', 'prob_expr_s', 'prob_feels_s']]
predictions = pd.concat([exp1_predictions, exp2_predictions], axis=1)
predictions.to_csv(args.output_file, sep="\t", index=False)
