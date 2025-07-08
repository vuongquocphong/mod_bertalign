# list `the file in sktmt/vi`
import gc
import os

import torch
from bertalign import Bertalign

# Get all file end with .par
files = [f for f in os.listdir('sktmt/zh') if f.endswith('.par')]

for file in files:
    
    with open(f'sktmt/zh/{file}', 'r', encoding='utf-8') as f:
        zh_lines = f.read()
    
    with open(f'sktmt/vi/{file}', 'r', encoding='utf-8') as f:
        vi_lines = f.read()
    
    aligner = Bertalign(src=zh_lines, tgt=vi_lines)
    aligner.align_sents()

    def create_bead( aligner ):
        for bead in aligner.result:
            src_line = aligner._get_line(bead[0], aligner.src_sents)
            tgt_line = aligner._get_line(bead[1], aligner.tgt_sents, ' ')
            # calculate similarity
            yield (src_line, tgt_line)

    with open(f'sktmt/{file}_alignments.txt', 'w', encoding='utf-8') as f:
        for alignment in create_bead(aligner):
            f.write(alignment[0] + "\t" + alignment[1] + "\n")
    
    # deallocate aligner and wait for garbage collection
    del aligner
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
