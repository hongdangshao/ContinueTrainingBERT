# ContinueTrainingBERT
Continue Training BERT with transformers  
Continue Training BERT in the vertical field
# Quickstart
### 1. Install transformers
`pip install transormers`
### 2. Prepare your data
**NOTICE : Your data should be prepared with two sentences in one line with tab(\t) separator**
```
This is the first sentence. \t This is the second sentence.\n
Continue Training \t BERT with transformers\n
```
### 3. Continue training bert
`python main.py`

 ---
# two models can be used
## 1.Using transformers model [BertForPreTraining](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.BertForPreTraining)
- inputs
  - input_ids              # [sentence0, sentence1] the original index based on the tokenizer
  - token_type_ids         # [0, 1] zero represent sentence0
  - attention_mask         # [1, 1] The areas that have been padded will be set to 0
  - labels                 # [....] masked, real index
  - next_sentence_label    # [0 or 1] zero represent sentence0 and sentence1 have no contextual relationship
  - ...
- outputs
  - loss                      # masked_lm_loss + next_sentence_loss, predict masked loss and next sentence loss  
  - prediction_logits
  - seq_relationship_logits
  - ...
## 2.Using transformers model [BertForMaskedLM](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/bert#transformers.BertForMaskedLM)
- inputs
    - input_ids
    - token_type_ids      # [1,1] unused
    - attention_mask      
    - labels
    - ...
- outputs
    - loss                # masked_lm_loss
    - logits              # prediction_score
