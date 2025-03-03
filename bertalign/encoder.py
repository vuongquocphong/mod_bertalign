import numpy as np

from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps

class Encoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name,  trust_remote_code=True)
        self.model_name = model_name

    def transform(self, sents, num_overlaps, lang='', ner_dict={}):
        tmp_sents = []
        for sent in sents:
            tmp_sents.append(sent)
        if lang == 'zh':
            for sent in tmp_sents:
                for key, value in ner_dict.items():
                    if key in sent:
                        sent = sent.replace(key, '<NER>')
        if lang == 'vi':
            for sent in tmp_sents:
                for key, value in ner_dict.items():
                    if value in sent:
                        sent = sent.replace(value, ' <NER> ')
        overlaps = []
        for line in yield_overlaps(tmp_sents, num_overlaps):
            overlaps.append(line)

        sent_vecs = self.model.encode(overlaps)
        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs
