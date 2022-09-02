## Chat_Basics
This repo covers Basic Models for Dialogue Generation Task.
The main purpose is to check the developments while comparing each model.
For a fairer comparision, some modifications are applied and as a result, some parts may differ from those in papers.

<br>

## Table of Contents
  * [Model desc](#model-desc)
  * [Configs](#configs)
  * [How to Use](#how-to-use)
  * [Results](#results)
  * [References](#references)

<br>


## Model desc

### [Sequence-to-Sequence](https://arxiv.org/abs/1409.3215)
> As the name **"Sequence-to-Sequence"** suggests, it is an end-to-end sequence model.
The Architecture consists of Encoder and Decoder. In detail, the Encoder first makes Contetx Vectors from Input Sequences. 
And then the Decoder gets Encoder Outputs and Auto Regressive Values from Target sequences as an Input Values to return Target Sequences.
Before Sequence-to-Sequence Architecture was generally applied to various NLP Tasks, Statistical Models outperformed Neural End-to-End models.
This Architecture has proved its significance by opening Neural end-to-end Model Era.

<br>

### [Attention Mechanism](https://arxiv.org/abs/1409.0473)
> The main idea of Attention Mechanism came from Human's Brain Cognition Process.
People live with a variety of information, but when faced with a specific problem, people usually focus on the information needed to solve the problem. We call this as an **Attention**.
The Architecture also use Encoder-Decoder architecture, but the difference is that the Decoder uses Attention Operation to make predictions.
By using Attention Mechanism, the model could avoid Bottle Neck problem, which results in Better performances in Quantative and Qualitive Evaluation at the same time.

<br>


### [Transformer](https://arxiv.org/abs/1706.03762)
> Natural Language is inevitably a time-series data. In order to consider the time series aspect, the RNN structure was considered as the only option.
But **Transformer** broke this conventional prejudice and showed remarkable achievements by only using Attention Mechanism without any RNN Layer.
Existing RNN models always had two chronic problems. First is a vanishing gradient problem which is apparent as the sequence length gets longer. Second is Recurrent Operation process itself, which makes parallel processing difficult.
But the Transformer solved these problems only with Attentions. As a result, the architecture not only performs well in a variety of NLP tasks, but is also fast in speed.

<br>


<br>
<br>

## Configs
### Model Configs

<table>
  <tr>
    <th>Seq2Seq</th>
    <th>Attention</th>
    <th>Transformer</th>
  </tr>
  <tr>
    <td>
      <strong>&nbsp; &centerdot; Input Dimension:</strong> 10000 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Output Dimension:</strong> 10000 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Embedding Dimension:</strong> 256 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Hidden Dimension:</strong> 512 &nbsp;<br> 
      <strong>&nbsp; &centerdot; N_Layers:</strong> 2 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Dropout Ratio:</strong> 0.5 &nbsp;
    </td>
    <td>
      <strong>&nbsp; &centerdot; Input Dimension:</strong> 10000 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Output Dimension:</strong> 10000 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Embedding Dimension:</strong> 256 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Hidden Dimension:</strong> 512 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Dropout Ratio:</strong> 0.5 &nbsp;
    </td>
    <td>
      <br>
      <strong>&nbsp; &centerdot; Input Dimension:</strong> 10000 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Output Dimension:</strong> 10000 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Embedding Dimension:</strong> 256 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Hidden Dimension:</strong> 256 &nbsp;<br> 
      <strong>&nbsp; &centerdot; PFF Dimension:</strong> 512 &nbsp;<br> 
      <strong>&nbsp; &centerdot; N_Layers:</strong> 3 &nbsp;<br> 
      <strong>&nbsp; &centerdot; N_Heads:</strong> 8 &nbsp;<br> 
      <strong>&nbsp; &centerdot; Dropout Ratio:</strong> 0.1 &nbsp;<br>
      <br>
    </td>
  </tr>
</table>

<br>

### Training Configs

* **Batch Size:** 128
* **Num of Epochs:** 10
* **Learning Rate:** 1e-4
* **Label Smoothing:** 0.1
* **Optimizer:** Adam Optimizer
* **Tokenization:** BPE Tokenziation
* **Loss Function:** Cross Entropy Loss
* **Data:** [Downsized Single-Turn Daily Dialogue Dataset](https://github.com/moon23k/NLP_datasets#dialogue)
* Applied Different Initialization for Each Models

<br>

<center>
  <img src="https://user-images.githubusercontent.com/71929682/168110116-374d3ac9-48d6-41e3-a2ce-d216f2e76422.png" width="70%" height="60%">
</center>


<br>
<br>

## How to Use
**First clone git repo in your env**
```
git clone https://github.com/moon23k/NMT_Basic
```

<br>

**Download and Process Dataset by the code below**
```
cd NMT_Basic
bash prepare_data.sh
```

<br>

**Train models with "train.py" file (scheduler is optional)**
```
python3 train.py -model ['seq2seq', 'attention', 'transformer'] -scheduler ['constant', 'noam', 'cosine_annealing_warm', 'exponential', 'step']
```

<br>

**Test trained models with "test.py" file**
```
python3 test.py -model ['seq2seq', 'attention', 'transformer']
```

<br>

**Test with user input sentence via trained models**
```
python3 inference.py -model ['seq2seq', 'attention', 'transformer']
```


<br>
<br>


## Results

**Loss**

<br>

**PPL**

<br>

**BLEU**

<br>
<br>

## References
* Sequence to Sequence Learning with Neural Networks
* Neural Machine Translation by Jointly Learning to Align and Translate
* Attention is all you need
