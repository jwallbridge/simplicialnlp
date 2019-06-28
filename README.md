# Simplicial Language Transformer

This is the code repository for the language component of the project [simplicialtransformer](https://github.com/dmurfet/simplicialtransformer) by James Clift, Dmitry Doryn, Daniel Murfet and James Wallbridge.

The repository includes :
* `notes-background.md` which details various theoretical motivations based on existing literature.
* `notes-implementation.md` with issues running the code and increasing efficiency.
* `notes-experiments.md` the experiment log.

The notebooks include (for n <= 3) :
* `trans_v1.ipynb` the vanilla transformer (Vaswani et al).
* `unitrans_v1.ipynb` the universal transformer (Dehghani et al).
* `simptrans_v1.ipynb` the simplicial transformer.
* `unisimptrans_v1.ipynb` the universal simplicial transformer.

## File content

* `data_download.py`   
Downloads and preprocesses the training and evaluation WMT datasets. After the data is downloaded and extracted, the training data is used to generate a vocabulary of subtokens. The evaluation and training strings are tokenized, and the resulting data is sharded, shuffled, and saved as TFRecords (tensorflow's binary storage format).

* `transformer_main.py`  
Creates a Transformer model, and trains it using Tensorflow Estimator. It contains
  * `dataset.py`  
  Contains batching scheme and shuffling.
  * `model_params.py`  
  Defines transformer model parameters.
  * `transformer.py`
  Defines the transformer model and uses 
    * `attention_layer.py`   
    * `beam_search.py`   
    * `embedding_layer.py`   
    * `ffn_layer.py`   
    * `model_utils.py` (position encoding, look-ahead-mask and padding mask)   
    * `tokenizer.py` (defines subtokenizer class to encode and decode strings)
  * `metrics.py`    
  Functions for calculating loss, accuracy, and other model metrics.
  * `schedule.py`  
  Abstract training on a step or epoch basis.
  
* `translate.py`  
Translate the model. Contains the script to use the trained model to translate input text or file. Each line in the file is translated separately.

* `compute_bleu.py`  
Script to compute official BLEU score.
 
