# Modified Bertalign Sentence Alignment Approach to support Vietnamese

## Introduction

This repository is a modified version of the original implementation of Bertalign to support Vietnamese.

## Requirements (MUST HAVE)

Python 3.10.11

## Installation

```bash
pip install -r requirements.txt
```

## Usage

- The format of a directory for alignment should have 2 files:

```
<data_dir_name>
├── "chinese_pars.txt"
├── "translation_pars.txt"
├── "chinese_snts.txt"
└── "translation_snts.txt"
```

- Then, in the `align_main.py` file, change the names list to all the directories you want to align. Then run the following command:

```bash
python3.10 align_main.py <dir> <top_k> <max_align> <type>
```

- In which:
    - dir: name of the directory that contains files needed to be aligned
    - top_k: used to find k nearest neighbors of the source sentences
    - max_align: the maximum sum of number of source sentences and target sentences, which determines the possible alignment types
    - type: if you have splitted the paragraph into sentences, choose `snts`, else, choose `pars`

- The output will be saved in the `data_name_dir` directory with the name `alignments.txt`

## References

[Bertalign](https://github.com/bfsujason/bertalign)
