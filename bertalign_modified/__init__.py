"""
Bertalign initialization
"""

__author__ = "Jason (bfsujason@163.com)"
__version__ = "1.1.0"

from bertalign.encoder import Encoder

# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html

model_name = "LaBSE" # Default model
# model_name = "all-MPNet-base-v2" # Best for general-purpose model
# model_name = "paraphrase-multilingual-mpnet-base-v2" # Another high-quality multilingual model
# model_name = "distiluse-base-multilingual-cased-v2" # Improved multilingual embeddings for semantic similarity.
model = Encoder(model_name)

from bertalign.aligner import Bertalign