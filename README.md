# STAT7008_Project

## Python environment

Virtual Environment:

```bash
conda create -n nlp_env python=3.10.9
conda activate nlp_env
```

Packages:

```bash
pip install -r requirements.txt
```

## Word Embedding

**~~不用word embedding，one-hot编码输入向量长度等于词库长度，效果可能不好？~~**

~~现成的embedding（未验证）：~~ https://sites.google.com/site/rmyeid/projects/polyglot

-  Word2Vec ✔️ (4)
- GloVe
- FastText
- Elmo
- Bert ✔️

## Corpus

- Indonesian (Bahasa): https://dumps.wikimedia.org/idwiki/
- Javanese: https://dumps.wikimedia.org/jvwiki/

## Machine Translation
- lstm ✔️
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