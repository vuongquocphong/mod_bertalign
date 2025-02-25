from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from bertalign import utils
import json

tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def combine_entities(ner_result):
    combined_entities = []
    current_entity = []
    current_tag = None
    start_idx = None
    end_idx = None

    for token in ner_result:
        tag = token['entity']
        word = token['word']

        if tag.startswith("B-"):
            # Save the previous entity if exists
            if current_entity:
                combined_entities.append({
                    "entity": " ".join(current_entity),
                    "type": current_tag,
                    "start": start_idx,
                    "end": end_idx
                })
            # Start a new entity
            current_entity = [word]
            current_tag = tag[2:]  # Remove "B-" prefix
            start_idx = token['start']
            end_idx = token['end']
        elif tag.startswith("I-") and current_tag == tag[2:]:
            # Continue the current entity
            current_entity.append(word)
            end_idx = token['end']
        else:
            # Save the previous entity if exists and reset
            if current_entity:
                combined_entities.append({
                    "entity": " ".join(current_entity),
                    "type": current_tag,
                    "start": start_idx,
                    "end": end_idx
                })
            current_entity = []
            current_tag = None
            start_idx = None
            end_idx = None

    # Save the last entity if exists
    if current_entity:
        combined_entities.append({
            "entity": " ".join(current_entity),
            "type": current_tag,
            "start": start_idx,
            "end": end_idx
        })

    return combined_entities

sents = []
data_dir = ['tqdn1']
for dir_name in data_dir:
    with open(f'{dir_name}/translation_pars.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sents += utils.split_sents(line.strip(), 'zh')

entities = []
for sent in sents:
    ner_result = nlp(sent)
    combined_entities = combine_entities(ner_result)
    entities.append(combined_entities)

with open('vn_ner_data.json', 'w', encoding='utf-8') as f:
    for i, item in enumerate(entities):
        json_data = {}
        json_data['text'] = sents[i]
        json_data['entities'] = item
        line = json.dumps(json_data, ensure_ascii=False)
        f.write(line + '\n')
