from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

from bertalign_modified import model
from bertalign_modified.corelib import *
from bertalign_modified.utils import *
from bertalign_modified.utils import convert_zh, convert_vn, convert_words_to_indexList

class Bertalign:
    def __init__(self,
                 src,
                 tgt,
                 model = model,
                 max_align=6,
                 top_k=3,
                 win=5,
                 skip=-0.1,
                 margin=True,
                 len_penalty=True,
                 sentence_num_penalty=True,
                 snt_num_pen_val=0.06,
                 union_score=True,
                 union_cor_val=0.6,
                 is_split=False,
               ):
        self.src = src
        self.model = model
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        self.sentence_num_penalty = sentence_num_penalty
        self.snt_num_pen_val = snt_num_pen_val
        self.union_score = union_score
        self.union_cor_val = union_cor_val
        
        src = clean_text(src)
        tgt = clean_text(tgt)
        src_lang = 'zh'
        tgt_lang = 'vi'
        
        # Split into sentences
        if is_split:
            src_sents = src.splitlines()
            tgt_sents = tgt.splitlines()
        else:
            src_sents = split_sents(src, src_lang)
            tgt_sents = split_sents(tgt, tgt_lang)

        # Get the number of sentences
        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        
        print("Source language: {}, Number of sentences: {}".format(src_lang, src_num))
        print("Target language: {}, Number of sentences: {}".format(tgt_lang, tgt_num))

        # Convert sentences into embeddings
        print("Embedding source and target text using {} ...".format(self.model.model_name))
        src_vecs, src_lens = self.model.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = self.model.transform(tgt_sents, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs

    def align_sents(self):

        # print("Preparing words list ...")
        converted_src, converted_tgt, src_word_len, tgt_word_len = self._prepare_words_list()

        # print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k)
        first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(self.src_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)

        # print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        # second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
        #                                     second_w, second_path, second_alignment_types,
        #                                     self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
        second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                            converted_src, converted_tgt, src_word_len, tgt_word_len,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty, sentence_num_penalty=self.sentence_num_penalty, snt_num_pen_val=self.snt_num_pen_val, union_score=self.union_score, union_cor_val=self.union_cor_val)
        second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)
        
        print("Finished! Successfully aligning {} {} sentences to {} {} sentences\n".format(self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
        self.result = second_alignment
    
    def print_sents(self):
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    def _prepare_words_list(self):

        # start_time = time.time()

        # Convert zh text to words list
        converted_src, src_word_len = convert_zh(self.src, self.max_align - 1)
        converted_zh_len = len(converted_src[0])

        # Prepare index dictionary of each words
        words_index = convert_words_to_indexList(converted_src, self.max_align - 1)

        # # Check whether the number of converted_src and src_num are equal
        if converted_zh_len != self.src_num:
            print("Error: The number of converted source sentences does not match the number of source sentences.")
            print("Converted source sentences: {}".format(converted_zh_len))
            print("Source sentences: {}".format(self.src_num))
            raise ValueError("The number of converted source sentences does not match the number of source sentences.")

        # Convert vn text to words list
        converted_tgt, tgt_word_len = convert_vn(self.tgt_sents, self.max_align - 1)

        # end_time = time.time()
        # print("Time taken to convert sentences: {:.2f} seconds".format(end_time - start_time))

        return words_index, converted_tgt, src_word_len, tgt_word_len

    @staticmethod
    def _get_line(bead, lines, split_char=''):
        line = ''
        if len(bead) > 0:
            line = split_char.join(lines[bead[0]:bead[-1]+1])
        return line