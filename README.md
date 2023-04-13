# SentBS: Sentence-level Beam Search for Controllable Summarization
<!-- **Authors**: Chenhui Shen, Liying Cheng, Lidong Bing, Yang You and Luo Si -->

This repository contains code and related resources of our paper ["SentBS: Sentence-level Beam Search for Controllable Summarization"](https://aclanthology.org/2022.emnlp-main.699/).

<!-- :star2: Check out this awesome [[demo]](https://huggingface.co/spaces/joaogante/contrastive_search_generation) generously supported by Huggingface ([@huggingface](https://github.com/huggingface) :hugs:) which compares contrastive search with other popular decoding methods. Many thanks to Huggingface :hugs:!  -->


****
If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@article{shen2022sentbs,
  title={SentBS: Sentence-level Beam Search for Controllable Summarization},
  author={Shen, Chenhui and Cheng, Liying and Bing, Lidong and You, Yang and Si, Luo},
  journal={EMNLP 2022},
  year={2022}
}

@article{shen2022mred,
  title={MReD: A Meta-Review Dataset for Structure-Controllable Text Generation},
  author={Shen, Chenhui and Cheng, Liying and Zhou, Ran and Bing, Lidong and You, Yang and Si, Luo},
  journal={Findings of ACL},
  year={2022}
}
```

<!-- ****

### News:
* [2022/10/26] The paper "Contrastive Search Is What You Need For Neural Text Generation" is publicly released!

**** -->

<span id='all_catelogue'/>

### Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#reproduce_examples'>2. Running our code</a>
    * <a href='#pre-requisites'>2.1. Pre-requisites</a>
    * <a href='#reproduce'>2.2. Commands to reproduce our results</a>
        * <a href='#sent-ctrl'>2.2.1. Reproduce sent-ctrl</a>
        * <a href='#classifier'>2.2.2. Train Classifier</a>
        * <a href='#sent-ctrl_sentbs'>2.2.3. Reproduce Sent-Ctrl + SentBS</a>
        * <a href='#seg-ctrl'>2.2.4. Reproduce Seg-Ctrl and Seg-Ctrl + SentBS </a>
    
****

<span id='introduction'/>

#### 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>

A wide range of control perspectives have been explored in controllable text generation. Structure-controlled summarization is recently proposed as a useful and interesting research direction. However, current structure-controlling methods have limited effectiveness in enforcing the desired structure. To address this limitation, we propose a sentence-level beam search generation method (SentBS), where evaluation is conducted throughout the generation process to select suitable sentences for subsequent generations. We experiment with different combinations of decoding methods to be used as subcomponents by SentBS and evaluate results on the structure-controlled dataset MReD. Experiments show that all explored combinations for SentBS can improve the agreement between the generated text and the desired structure, with the best method significantly reducing the structural discrepancies suffered by the existing model, by approximately 68%.

****


<span id='reproduce_examples'/>


#### 2. Running our Code


<span id='pre-requisites'/>

##### 2.1. Pre-requisites: <a href='#all_catelogue'>[Back to Top]</a>

For our code, we use the Huggingface Transformers of version 4.16.2. 
To install a specific verison of transformers, check out <a href="https://github.com/huggingface/transformers/blob/main/examples/README.md">here</a>.

For the Bert-Score metric, we follow <a href="https://github.com/Tiiiger/bert_score"> this repository</a>.

We use the <a href="https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct."> public check point of Bart-Large </a> pretrained on CNN/DM as our base architecture.

##### 2.2. Commands to reproduce our results: <a href='#all_catelogue'>[Back to Top]</a>

For all experiments below, please download our processed data from <a href="https://drive.google.com/file/d/1U6EPGuFyTG6ZMsqIrHQb6pjoyZk_EMux/view?usp=sharing">here</a>.
Unzip the downloaded data and place all data folders under the root folder named ```/data```.

<span id='sent-ctrl'/>

###### 2.2.1. Reproduce Sent-Ctrl (Table 1 upper section): <a href='#all_catelogue'>[Back to Top]</a>

We include the <a href="https://drive.google.com/file/d/1U6EPGuFyTG6ZMsqIrHQb6pjoyZk_EMux/view?usp=sharing">reformatted data</a> used for our experiments. The original data can also be obtained from <a href="https://github.com/Shen-Chenhui/MReD/tree/master/summarization/abstractive/filtered_controlled_data">here</a>. 

To reproduce the sent-ctrl baseline, run:

```yaml
CUDA_VISIBLE_DEVICES=0 python ctrl_transformer.py --model_name_or_path facebook/bart-large-cnn --do_train --do_eval --do_predict --train_file data/original_clean/train_rate_concat_sent-ctrl.csv --validation_file data/original_clean/val_rate_concat_sent-ctrl.csv --test_file data/original_clean/test_rate_concat_sent-ctrl.csv --output_dir ./results/sentctrl_reproduced  --seed 0 --save_total_limit 3 --gen_target_max 800 --predict_with_generate --eval_steps 500 --max_source_length 2048
```

<span id='classifier'/>

###### 2.2.2. Train Classifier: <a href='#all_catelogue'>[Back to Top]</a>
For the MReD dataset, we additionally train a sentence classifier so that during generation, the selection of sentence options is based on both the category classification score as well as the sequence likelihood.

The classifier is trained on the <a href="https://drive.google.com/file/d/1U6EPGuFyTG6ZMsqIrHQb6pjoyZk_EMux/view?usp=sharing">LSTM-labelled training data split</a>.

The base architecture used for the classifier is the huggingface <a href="https://huggingface.co/roberta-large">Roberta-Large</a> model.
```yaml
CUDA_VISIBLE_DEVICES=0 python train_sent_classifier.py --model_path roberta-large
```

<span id='sent-ctrl_sentbs'/>

###### 2.2.3. Reproduce Sent-Ctrl + SentBS (Table 1 upper section): <a href='#all_catelogue'>[Back to Top]</a>

For the following commands, you may adjust the ```k``` value with the flag ```--gen_size```.
* For nucleus sampling:
```yaml
CUDA_VISIBLE_DEVICES=0 python beam_search_sent.py --gen_size 8 --beam_size 4 --top_p 0.9 --res_dir results/sampling --generation_model_path results/sentctrl_reproduced --test_file data/original_clean/test_rate_concat_sent-ctrl.csv --gen_mode sample --write --eval_rouge --load_classifier --classification_model_path <path_to_classification_model>
```

 <!-- 
 --generation_model_path ../ctrl-transformer/results/original_clean_extra_tokens/
 --classification_model_path /mnt/workspace/project/ecpe_transformer/mred_sentence_classification/roberta-large/
 --model_name_or_path /mnt/workspace/utils/huggingface_models/bart-large-cnn 
 -->

* For beam sampling:
```yaml
CUDA_VISIBLE_DEVICES=0 python beam_search_sent.py --gen_size 8 --beam_size 4 --top_p 0.9 --res_dir results/beam_sampling --generation_model_path results/sentctrl_reproduced --test_file data/original_clean/test_rate_concat_sent-ctrl.csv --gen_mode beam_sample --write --eval_rouge --load_classifier --classification_model_path <path_to_classification_model>
```

* For beam search + nucleus sampling
```yaml
CUDA_VISIBLE_DEVICES=0 python beam_search_sent.py --gen_size 8 --beam_size 4 --top_p 0.9 --res_dir results/mixed_bs_ns --generation_model_path results/sentctrl_reproduced --test_file data/original_clean/test_rate_concat_sent-ctrl.csv --gen_mode beam_search_sent --write --eval_rouge --load_classifier --classification_model_path <path_to_classification_model>
```


* For beam search + beam sampling + nucleus sampling

```yaml
CUDA_VISIBLE_DEVICES=0 python beam_search_sent.py --gen_size 8 --beam_size 4 --top_p 0.9 --res_dir results/mixed_all --generation_model_path results/sentctrl_reproduced --test_file data/original_clean/test_rate_concat_sent-ctrl.csv --gen_mode beam_search_sent  --beam_sample --write --eval_rouge --load_classifier --num_beam_sample_gen 4 --classification_model_path <path_to_classification_model>
```
You may use the flag ```--num_beam_sample_gen``` to control the number of sentencens generated by beam sampling. 


<span id='seg-ctrl'/>

###### 2.2.4. Reproduce Seg-Ctrl and Seg-Ctrl + SentBS (Table 1 bottom section): <a href='#all_catelogue'>[Back to Top]</a>

To reproduce the seg-ctrl baseline, run:

```yaml
CUDA_VISIBLE_DEVICES=0 python ctrl_transformer.py --model_name_or_path facebook/bart-large-cnn --do_train --do_eval --do_predict --train_file data/original_seg_clean/train.csv --validation_file data/original_seg_clean/val.csv --test_file data/original_seg_clean/test.csv --output_dir results/segctrl_reproduced  --seed 0 --save_total_limit 3 --gen_target_max 800 --predict_with_generate --eval_steps 500 --max_source_length 2048
```

For seg-ctrl+SentBS, run

```yaml
CUDA_VISIBLE_DEVICES=0 python segctrl_sentbs.py --res_dir results/segctrl_sentbs --generation_model_path results/segctrl_reproduced --test_file data/original_seg_clean/test.csv --gen_mode beam_search_sent --load_classifier --classification_model_path ../ecpe_transformer/mred_sentence_classification/roberta-large/ --gen_size 8 --beam_size 4 --beam_sample --eval_rouge --run_num 0 --write
```
