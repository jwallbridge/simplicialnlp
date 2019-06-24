This component of the project involves training the simplicial transformer on a common machine translation task WMT14 EN-DE.  For debugging the architecture, we use IWSLT14 EN-DE which is significantly smaller.  If time permits, WMT14 EN-FR can be tackled.

For this task, we expect the simplicial transformer to encode more refined representations of N-grams for N > 1. This would provide an improved architectural block, in comparison to existing 1-simplex attention, for machine translation.  Indeed, the prominent measure for evaluating translations, the BLEU-score, uses a 4-gram precision comparison algorithm.  Therefore, we expect that a 3-truncated simplicial attention block to be sufficient for improving existing benchmarks.

For WMT14 DE-EN, these BLEU-benchmarks are  
* Transformer (base) : 27.3
* Transformer (big) : 28.4
* Universal Transformer (base) : 28.9  

for 8 x P100 GPUs.  Note that the UT result does not use ACT.

## Current state-of-the-art

* https://github.com/sebastianruder/NLP-progress  
This monitors the progress on common NLP tasks.

* https://github.com/sebastianruder/NLP-progress/blob/master/english/machine_translation.md  
In particular for WMT 2014 EN-DE and EN-FR.

We make some comments on the top performing architecture (DeepL notwithstanding) which is Transformer based (Edunov et al). This outperforms both the previous state-of-the-art Transformer model (Ott et al) which was a refined scaled Big Transformer 
(128 GPUs) and the state-of-the-art non-Transformer based model (We et al) based on dynamic convolutions.

The model of Edunov et al is also Big (128 GPUs) but uses a back-translation component from (Sennrich et al).  
Back-translation first trains an intermediate system on parallel data (bilingual data and monolingual target data) 
which is used to translate the target monolingual data into the source language.  The result is a parallel corpus where 
the source side is *synthetic* MT output while the target is genuine human written text.  The synthetic parallel 
corpus is then added to the real bitext in order to train a final system that will translate from the source to the 
target language.  

## Data and pre-preprocessing

* https://github.com/pytorch/fairseq/tree/master/examples/translation  
This includes IWSLT 2014 (lightweight task) and WMT 2014 (heavyweight task).

* https://github.com/tensorflow/models/tree/master/official/transformer  
This is for WMT 2017 and contains alot of useful information.


## Embedding Layer

Keras offers an Embedding layer that can be used for neural networks on text data.  It requires that the input data be integer encoded, so that each word is represented by a unique integer. This data preparation step can be performed using the Tokenizer API also provided with Keras.

The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset.  It is defined as the first hidden layer of a network and must specify 3 arguments:
* *input\_dim*: This is the size of the vocabulary in the text data. For example, if the data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
* *output\_dim*: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
* *input\_length*: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000.


## Adaptive computational time

From Dehghani et al 
>In sequence processing systems, certain symbols (e.g. some words or phonemes) are usually more ambiguous than others. It is therefore reasonable to allocate more processing resources to these more ambiguous symbols. Adaptive Computation Time (ACT) (Graves, 2016) is a mechanism for dynamically modulating the number of computational steps needed to process each input symbol (called the “ponder time”) in standard recurrent neural networks based on a scalar halting probability predicted by the model at each step.  

>Inspired by the interpretation of Universal Transformers as applying self-attentive RNNs in parallel to all positions in the sequence, we also add a dynamic ACT halting mechanism to each position (i.e. to each per-symbol self-attentive RNN). Once the per-symbol recurrent block halts, its state is simply copied to the next step until all blocks halt, or we reach a maximum number of steps. The final output of the encoder is then the final layer of representations produced in this way.

Note that when we are minimizing the loss, we are actually maximizing the halting probabilities, so we are maximizing the probability to stop - this makes the network stop earlier.


## Beam search

Beam search is used during decoding to find the sequence that maximizes a score function given a trained model.  The beam size (beam size 1 corresponds to greedy search, ie. argmax).  

Two refinements to the pure max-probability based beam search algorithm are a coverage penalty and length normalization. With length normalization, the aim is to account for the fact that we have to compare hypotheses of different length. Without some form of length-normalization, regular beam search will favour shorter results over longer ones on average since a negative log-probability is added at each step, yielding lower (more negative) scores for longer sentences. 


## Probing, sparsity and visualization

Recent work (Voita et al) (Michel et al) show that a large percentage of attention heads can be pruned away without significantly impacting performance.

From Clarke et al
>We compute the average entropy of each head's attention distribution. We find that some attention heads, especially in lower layers, have very broad attention. These high-entropy attention heads typically spend at most 10% of their attention mass on any single word. The output of these heads is roughly a bag-of-vectors representation of the sentence.

## Model comparison

For N-gram difference analysis, word accuracy and many other comparison tools, see
* https://github.com/neulab/compare-mt



