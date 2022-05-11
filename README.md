# Central Language-aware Layer
This is the codes & scripts of paper Adapting to Non-Centered Languages for Zero-shot Multilingual Translation  
There are two types of proposed method as following,
## Full CLL (FCLL)
Namely, all decoder layers of Transformer have constructed CLL.
## Single-Disentangled CLL (SD)
Inspired by the work of [Liu et al. 2021](https://aclanthology.org/2021.acl-long.101.pdf)  
We remove the residual connection of FFN in a middle encoder layer to weaken the linguistic features of encoding.  
To keep the balance between encoding and decoding, we empirically build CLL in the middle decoder.  
Specifically, given $N$ encoder and decoder layers of the transformer, we remove the residual connection of the FFN in the encoder and replace the FFN with CLL in the decoder at $N/2+1$th layer of both networks.
  
Examples:  
$N=5$, layer index of removing residual connection and building CLL is 3;  
$N=6$, layer index of removing residual connection and building CLL is 4;  
## Structure
```
| -- CLL  
  | -- data  
    | -- __init__.py  
    | -- language_pair_dataset_cll.py
  | -- models  
    | -- __init__.py  
    | -- transformer_cll.py  
    | -- transformer_decoder_cll.py  
    | -- transformer_encoder_cll.py  
  | -- modules  
    | -- __init__.py  
    | -- transformer_layer_cll.py  
  | -- task
    | -- __init__.py  
    | -- ......
  | -- __init__.py  
| -- SD  
| -- CLL_merged  
| -- SD_merged  
| -- directionFiles  
  | -- iwslt  
    | -- train_src.txt  
    | -- train_tgt.txt  
    | -- valid_src.txt  
    | -- ......  
  | -- square  
  | -- triangle  
  | -- opus  
```
Note1: 'merged' means merge all dataset of different languages together based on prior works.  
We use 'merge' metric for handling data in IWSLT, Europarl, and OPUS-100.  
We use multilingual translation based on different languages pairs in TED.  
  
Note2: you need to change the hyperparameters including layer number, inner size of FNN, and url of reading direction files of merged data in the codes.
## Dataset
### IWSLT17
We take IWSLT17 from [MMCR4NLP](https://arxiv.org/abs/1710.01025)  
We merge all dataset of different languages together.
### OPUS-100
We merge all dataset of different languages together as [Zhang et al. 2020](https://aclanthology.org/2020.acl-main.148.pdf).
### Square
We take Europarl from [MMCR4NLP](https://arxiv.org/abs/1710.01025)  
We merge all dataset of different languages together.
### Triangle
We take Europarl from [MMCR4NLP](https://arxiv.org/abs/1710.01025)  
We set random seed as 1, and use np.random.shuffle to then subsample top 200K sentences in each language. 
### TED talk
We follow [Phillip](https://aclanthology.org/2020.emnlp-main.361.pdf) to use top 20 languages of TED.  
And, we do not use 'merge' strategy in TED.
## Scripts
```
#preprocessing:
fairseq-preprocess --joined-dictionary --source-lang input --target-lang output \
--trainpref iwslt/train \
--validpref iwslt/valid \
--testpref iwslt/test \
--destdir iwslt-bin/baseline

#running:
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup fairseq-train iwlst-bin/baseline \
--user-dir cll_merged \
--task translation_cll_merged --seed 1 \
--source-lang input --target-lang output \
--arch transformer_cll_merged --dropout 0.3 --language_num 4 \
--ddp-backend=no_c10d \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--share-all-embeddings \
--max-update 100000 \
--max-tokens 4000 --weight-decay 0.0001 \
--save-dir check_iwslt/1 \
--no-epoch-checkpoints \
--no-progress-bar \
--log-interval 200 > iwslt/1.log

#generate:
fairseq-generate iwlst-bin/baseline --gen-subset test \
--user dir cll_merged \
--task translation_cll_merged \
--path check_iwslt/1/checkpoint_best.pt \
--beam 4 --remove-bpe sentencepiece > iwslt/predict/seed_1/test.txt

#post-preprocessing:
# employ moses(https://github.com/moses-smt/mosesdecoder) to detokenize
# calculate results by sacrebleu (https://github.com/mjpost/sacreBLEU)

```
