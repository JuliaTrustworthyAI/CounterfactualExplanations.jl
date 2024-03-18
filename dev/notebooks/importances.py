import os
import json

from transformers import AutoTokenizer, RobertaForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

def extract_importances(model_path, input_strings):
    model = RobertaForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("gtfintechlab/FOMC-RoBERTa")

    scorer = SequenceClassificationExplainer(model, tokenizer, attribution_type='lig')
    
    attributions = []
    for t in input_strings:
        attributions.append(scorer(t, index=0, internal_batch_size=1))

    with open('temp/attributions.json', 'w') as f:
        f.write(json.dumps(attributions))