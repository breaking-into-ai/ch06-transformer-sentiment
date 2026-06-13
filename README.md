# Chapter 6 — Deep Learning: Neural Nets, Transformers, and a Sentiment Classifier

Companion code for **Breaking Into AI**, Chapter 6: *Deep Learning — Neural Networks, CNNs, RNNs, and Transformers*.

- `attention_from_scratch.py` — self-attention, multi-head attention, and a full transformer block in plain PyTorch, plus the FlickSageNet feed-forward network
- `finetune_sentiment.py` — fine-tune DistilBERT on IMDB movie-review sentiment (GPU recommended; Google Colab free tier works)
- `transfer_learning_resnet.py` — transfer learning on a pre-trained ResNet for poster genre classification

## Setup
```bash
pip install -r requirements.txt
python attention_from_scratch.py   # runs on CPU in seconds
```
