## Sum_Basics
This repo covers Basic Models for Abstractive Text Summarization Task.
The main purpose is to check the developments while comparing each model.
For a fairer comparision, some modifications are applied and as a result, some parts may differ from those in papers.
Compared to other tasks, summarization data is long. Therefore, the main purpose is to identify and compare the ability of basic models to handle long sentences.

<br>

### Table of Contents
> **[Model desc](#model-desc)** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **[Configurations](#configurations)** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **[Results](#results)** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **[How to Use](#how-to-use)** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **[References](#references)**
</br>
</br>


## Model desc

**[Sequence-to-Sequence](https://arxiv.org/abs/1409.3215)**
> As the name **"Sequence-to-Sequence"** suggests, it is an end-to-end sequence model.
The Architecture consists of Encoder and Decoder. In detail, the Encoder first makes Contetx Vectors from Input Sequences. 
And then the Decoder gets Encoder Outputs and Auto Regressive Values from Target sequences as an Input Values to return Target Sequences.
Before Sequence-to-Sequence Architecture was generally applied to various NLP Tasks, Statistical Models outperformed Neural End-to-End models.
This Architecture has proved its significance by opening Neural end-to-end Model Era.

<br>

**[Attention Mechanism](https://arxiv.org/abs/1409.0473)**
> The main idea of Attention Mechanism came from Human's Brain Cognition Process.
People live with a variety of information, but when faced with a specific problem, people usually focus on the information needed to solve the problem. We call this as an **Attention**.
The Architecture also use Encoder-Decoder architecture, but the difference is that the Decoder uses Attention Operation to make predictions.
By using Attention Mechanism, the model could avoid Bottle Neck problem, which results in Better performances in Quantative and Qualitive Evaluation at the same time.

<br>


**[Transformer](https://arxiv.org/abs/1706.03762)**
> Natural Language is inevitably a time-series data. In order to consider the time series aspect, the RNN structure was considered as the only option.
But **Transformer** broke this conventional prejudice and showed remarkable achievements by only using Attention Mechanism without any RNN Layer.
Existing RNN models always had two chronic problems. First is a vanishing gradient problem which is apparent as the sequence length gets longer. Second is Recurrent Operation process itself, which makes parallel processing difficult.
But the Transformer solved these problems only with Attentions. As a result, the architecture not only performs well in a variety of NLP tasks, but is also fast in speed.

<br>
<br>

## Configurations

> **Model Configs**

|  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `Seq2Seq` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `Attention` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `Transformer` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| :--- | :---: | :---: | :---: |
| **`Input Dimension`** | 10,000 | 10,000 | 10,000 |
| **`Output Dimension`** | 10,000 | 10,000 | 10,000 |
| **`Embedding Dimension`** | 256 | 256 | 256 |
| **`Hidden Dimension`** | 512 | 512 | 512 |
| **`PFF Dimension`** | - | - | 1024 |
| **`N Layers`** | 2 | - | 3 |
| **`N Heads`** | - | - | 8 |
| **`Dropout Ratio`** | 0.5 | 0.5 | 0.1 |

<br>
<br>

> **Training Configs**

* **Batch Size:** 128 </br>
* **Num of Epochs:** 10 </br>
* **Learning Rate:** 1e-4 </br>
* **Label Smoothing:** 0.1 </br>
* **Optimizer:** Adam Optimizer </br>
* **Tokenization:** BPE Tokenziation </br>
* **Loss Function:** Cross Entropy Loss </br>
* **Data:** CNN_Daily Mail </br>
* Applied Different Initialization for Each Models

<br>

<center>
  <img src="https://user-images.githubusercontent.com/71929682/168110116-374d3ac9-48d6-41e3-a2ce-d216f2e76422.png" width="80%" height="60%">
</center>

<br>
<br>

## Results

<center>
  <img src="https://user-images.githubusercontent.com/71929682/188311738-ec98d7c4-f10c-4b5d-aa8a-c538c58d974b.png" width="90%" height="70%">
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

## References
* Sequence to Sequence Learning with Neural Networks
* Neural Machine Translation by Jointly Learning to Align and Translate
* Attention is all you need
