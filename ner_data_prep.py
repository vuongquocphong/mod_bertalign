from bertalign import utils
import json

dir_list = ['tqdn1']

for dir_name in dir_list:
    res = []
    count = 0
    with open(f'{dir_name}/chinese_pars.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sents = utils.split_sents(line.strip(), 'zh')
            for sent in sents:
                json_data = {}
                json_data['id'] = count
                json_data['text'] = sent.strip()
                res.append(json_data)
                count += 1
    with open(f'{dir_name}/chinese_ner_data.json', 'w', encoding='utf-8') as f:
        for item in res:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')
    res = []
    count = 0
    with open(f'{dir_name}/translation_pars.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sents = utils.split_sents(line.strip(), 'zh')
            for sent in sents:
                json_data = {}
                json_data['id'] = count
                json_data['text'] = sent.strip()
                res.append(json_data)
                count += 1
    with open(f'{dir_name}/translation_ner_data.json', 'w', encoding='utf-8') as f:
        for item in res:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')
    print(f'{dir_name} done')