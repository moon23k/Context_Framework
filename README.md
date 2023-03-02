## Sum BERT

BERT is widely used in many NLP DownStreaming Transfer Learning. 
Since the BERT has pre-trained on MLM and NSP objectives, it is not straightforward to transfer to Sequence Genreration Task.
This is especially true when dealing with long sequences, such as Text Summarization.
To mend this problem **BERTSUM** has been proposed. But BERTSUM limits the sentence length to 512 and still does not completely solve the problem caused by the sequence length.
This repo deals with a series of study to solve this problem by mixing the concept of BERTSUM's Hierarchical Encoder and the concept of BERT fused Architecture, which has proposed in **Incorporating BERT into Neural Machine Translation** paper.

<br><br>

## Strategies

**Hierarchical Encoder** <br>

<br>

**Simple Enc-Dec Architecture** <br>

<br>

**Fused Enc-Dec Architecture** <br>

<br>

Hierarchical Encoder is used as Encoder, and two Encoder-Decoder connection methods are used to explore which method is effective.

<br><br>

## Experimental Setup

<br><br>

## Result

<br><br>

## How to use
```
git clone 
```

```
python3 setup.py
```

```
python3 run.py -mode [train, test, inference] -strategy [simple, fused]
```

<br><br>

## Reference
[**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805)

[**Text Summarization with Pretrained Encoders**](https://arxiv.org/abs/1908.08345)

[**Incorporating BERT into Neural Machine Translation**](https://arxiv.org/abs/2002.06823)
<br>
