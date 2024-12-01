# Modified Bertalign Sentence Alignment Approach to support Vietnamese

## Introduction

This repository is a modified version of the original implementation of Bertalign to support Vietnamese.

## Requirements

Python 3.10.15

## Installation

```bash
pip install -r requirements.txt
```

## Usage

- The format of a directory for alignment should have 2 files:
```
<data_dir_name>
├── "chinese_pars.txt"
└── "vietnamese_pars.txt"
```

- Then, in the `align_main.py` file, change the names list to all the directories you want to align. Then run the following command:
```bash
python3.10 align_main.py
```

- The output will be saved in the `data_name_dir` directory with the name `alignments.txt`

## References

[Bertalign](https://github.com/bfsujason/bertalign)
