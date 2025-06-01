import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from bertalign_modified.utils import yield_overlaps
import gc

class Encoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def transform(self, sents, num_overlaps, batch_size=32):
        """
        Transform sentences with batched encoding to prevent memory overflow.
        
        Args:
            sents: List of sentences
            num_overlaps: Number of overlap layers
            batch_size: Maximum number of sentences to encode in each batch
        """
        # Collect all overlaps first
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)
        
        total_overlaps = len(overlaps)
        print(f"Total overlaps to encode: {total_overlaps}")
        
        # Initialize result containers
        all_embeddings = []
        
        # Process in batches to avoid memory overflow
        for i in range(0, total_overlaps, batch_size):
            batch_end = min(i + batch_size, total_overlaps)
            batch_overlaps = overlaps[i:batch_end]
            
            print(f"Encoding batch {i//batch_size + 1}/{(total_overlaps + batch_size - 1)//batch_size} "
                  f"({len(batch_overlaps)} sentences)")
            
            # Encode this batch
            batch_vecs = self.model.encode(batch_overlaps)
            all_embeddings.append(batch_vecs)
            
            # Force garbage collection and clear GPU cache after each batch
            del batch_vecs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        sent_vecs = np.concatenate(all_embeddings, axis=0)
        
        # Clean up intermediate results
        del all_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reshape to expected format
        embedding_dim = sent_vecs.shape[1]
        sent_vecs = sent_vecs.reshape(num_overlaps, len(sents), embedding_dim)

        # Calculate length vectors
        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs = len_vecs.reshape(num_overlaps, len(sents))

        return sent_vecs, len_vecs