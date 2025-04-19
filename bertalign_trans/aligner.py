import numpy as np

from bertalign_trans import model, translator
from bertalign_trans.corelib import *
from bertalign_trans.utils import *

class Bertalign:
    def __init__(self,
                 src,
                 translated_src,
                 tgt,
                 model = model,
                 max_align=5,
                 top_k=3,
                 win=5,
                 skip=-0.1,
                 alpha=1.0,
                 margin=True,
                 len_penalty=True,
                 ner_penalty=True,
                 is_split=False,
                 ner_dict=None
               ):
        self.model = model
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.alpha = alpha
        self.margin = margin
        self.len_penalty = len_penalty
        self.ner_penalty = ner_penalty
        
        # Clean the source and target text
        src = clean_text(src)
        tgt = clean_text(tgt)

        # Set the source and target languages
        src_lang = 'zh'
        tgt_lang = 'vi'
        
        # Split sentences in both source and target text
        if is_split:
            src_sents = src.splitlines()
            translated_src_sents = translated_src.splitlines() 
            tgt_sents = tgt.splitlines()
        else:
            src_sents = split_sents(src, src_lang)
            tgt_sents = split_sents(tgt, tgt_lang)

        # Get number of sentences in source and target text
        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        
        # Detect language of source and target text
        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]
        
        print("Source language: {}, Number of sentences: {}".format(src_lang, src_num))
        print("Target language: {}, Number of sentences: {}".format(tgt_lang, tgt_num))

        # Transform sentences into vectors using the sentence embedding model
        print("Embedding source and target text using {} ...".format(self.model.model_name))
        src_vecs, src_lens = self.model.transform(src_sents, max_align - 1)
        translated_src_vecs, translated_src_lens = self.model.transform(translated_src_sents, max_align - 1)
        tgt_vecs, tgt_lens = self.model.transform(tgt_sents, max_align - 1)

        # Extract NER entities from source and target text
        src_ner = [[[] for _ in range(src_num)] for _ in range(self.max_align - 1)]
        tgt_ner = [[[] for _ in range(tgt_num)] for _ in range(self.max_align - 1)]
        ########################################################################

        # Length ratio for further length penalty
        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])


        # Update the class attributes
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.translated_src_sents = translated_src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.translated_src_lens = translated_src_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.translated_src_vecs = translated_src_vecs
        self.tgt_vecs = tgt_vecs
        
        # New NER attributes
        self.src_ner = src_ner
        self.tgt_ner = tgt_ner

    def align_sents(self):
        
        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.translated_src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k)
        first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)
        
        print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        # second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
        #                                     second_w, second_path, second_alignment_types,
        #                                     self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
        second_pointers = second_pass_align(self.translated_src_vecs, self.tgt_vecs, self.src_ner,  self.tgt_ner, self.translated_src_lens, self.tgt_lens,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, self.alpha, margin=self.margin, len_penalty=self.len_penalty, ner_penalty=self.ner_penalty)
        second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)
        
        print("Finished! Successfully aligning {} {} sentences to {} {} sentences\n".format(self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
        self.result = second_alignment
    
    def print_sents(self):
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line
