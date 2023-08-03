# ContinueTrainingBERT
Continue Training BERT with transformers  
Continue Training BERT in the vertical field  
This repository is just a simple example of bert pre-training   
##🎉Welcome everyone to improve this repository with me 🎉
## Roadmap
- [x] load pretrained weight
- [x] continue training
  - [x] Using transformers DataCollator class
  - [x] Using transformers Tokenizer class
  - [x] Using transformers Model class
  - [ ] Using transformers Trainer class
- [ ] Implement tokenizer class
- [ ] Implement bert model structure (class)
  - [ ] Implement bert embedding、encoder and pooler structure

---
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

---
# reference
- https://github.com/huggingface/transformers
- https://github.com/codertimo/BERT-pytorch
- https://github.com/wzzzd/pretrain_bert_with_maskLM
- https://github.com/circlePi/Pretraining-Yourself-Bert-From-Scratch