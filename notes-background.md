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






## Adaptive computational time

From Dehghani et al 
>In sequence processing systems, certain symbols (e.g. some words or phonemes) are usually more ambiguous than others. It is therefore reasonable to allocate more processing resources to these more ambiguous symbols. Adaptive Computation Time (ACT) (Graves, 2016) is a mechanism for dynamically modulating the number of computational steps needed to process each input symbol (called the “ponder time”) in standard recurrent neural networks based on a scalar halting probability predicted by the model at each step.  

>Inspired by the interpretation of Universal Transformers as applying self-attentive RNNs in parallel to all positions in the sequence, we also add a dynamic ACT halting mechanism to each position (i.e. to each per-symbol self-attentive RNN). Once the per-symbol recurrent block halts, its state is simply copied to the next step until all blocks halt, or we reach a maximum number of steps. The final output of the encoder is then the final layer of representations produced in this way.



