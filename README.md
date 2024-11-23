# STAT7008_Project

## Python environment

Virtual Environment:

```bash
conda create -n nlp_env python=3.10.9
conda activate nlp_env
```

Install Packages:

```bash
pip install -r requirements.txt
```

## Load the Datasets

```bash
git clone https://github.com/IndoNLP/nusax.git
```

## Download the Tokenizer

```bash
cd src/utils
python preprocess.py
```

## Word Embedding

- nn.embedding ✔️ (linear map)
- Word2Vec ✔️ (4)
- GloVe
- FastText
- Elmo
- Bert ✔️

## Corpus

- Indonesian (Bahasa): https://dumps.wikimedia.org/idwiki/
- Javanese: https://dumps.wikimedia.org/jvwiki/

## Machine Translation
- seq2seq based on lstm ✔️
- transformer ✔️
- TinyBert **?**

## 问题

**如何在少量数据上进行模型训练？**

## Pretrain Models

**Indonesian Model**

indobert: https://huggingface.co/indolem/indobert-base-uncased

**Multilingual Models**

mbert: https://huggingface.co/google-bert/bert-base-multilingual-cased

XLM-RoBERTa: https://huggingface.co/FacebookAI/xlm-roberta-base

**for Javanese, directly use \ fine-tune -- mbert \ XLM-RoBERTa?**