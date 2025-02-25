import re
from underthesea import sent_tokenize

def zh_pars_to_snts(zh_pars):
    limit = 1000
    sent_list = []
    zh_pars = re.sub('(?P<quotation_mark>([。.？！](?![”’"」\'）])))', r'\g<quotation_mark>\n', zh_pars)
    zh_pars = re.sub('(?P<quotation_mark>([。.？！]|…{1,2})[”’"」\'）])', r'\g<quotation_mark>\n', zh_pars)
    sent_list_ori = zh_pars.splitlines()
    for sent in sent_list_ori:
        sent = sent.strip()
        if not sent:
            continue
        else:
            while len(sent) > limit:
                temp = sent[0:limit]
                sent_list.append(temp)
                sent = sent[limit:]
            sent_list.append(sent)
    return sent_list

def vi_pars_to_snts(vi_pars):
    sent_list = sent_tokenize(vi_pars)
    return sent_list

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python pars2snts.py <input_file> <output_file> <lang>')
        sys.exit(1)
    list_snts = []
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        pars = f.readlines()
    if sys.argv[3] == 'zh':
        for par in pars:
            snts = zh_pars_to_snts(par)
            list_snts.extend(snts)
    elif sys.argv[3] == 'vi':
        for par in pars:
            snts = vi_pars_to_snts(par)
            list_snts.extend(snts)
    else:
        print('Language not supported yet.')
        sys.exit(1)
    with open(sys.argv[2], 'w', encoding='utf-8') as f:
        for snt in list_snts:
            f.write(snt + '\n')