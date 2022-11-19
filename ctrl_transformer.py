import json
import random
from pathlib import Path
from typing import Tuple, List
import argparse
from tqdm import tqdm

import pandas as pd
import torch
from pydantic import BaseModel
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BeamSearchScorer,
)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.generation_utils import (
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
)
from transformers import (
    pipeline,
    Pipeline,
    IntervalStrategy,
    Seq2SeqTrainingArguments,
)
from utils import (
    beam_search,
    beam_search_sent,
    get_prompts_from_input_text,
    postprocess_text,
    compute_metrics,
    get_logits_processor,
)

import run_summarization
from datasets import load_metric
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import os
import re # SentBS: post-process labels

os.environ["WANDB_DISABLED"] = "true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_beam_search(
        text: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        loss_fn=CrossEntropyLoss(),
        vocab_text: str = None,
        num_beams: int = 4,
        max_source_length: int = 2048,
        gen_target_min: int = 20,
        gen_target_max: int = 400,
):
    device = model.device
    # switch to evaluation mode
    model.eval()
    # prepare source
    encoder_input_ids = tokenizer(text,max_length=max_source_length,padding=False,truncation=True,return_tensors="pt").input_ids.to(device)
    encoder_outputs = model.get_encoder()(encoder_input_ids, return_dict=True)
    expanded_return_idx = (torch.arange(encoder_input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1))
    # instead of copy all states, only copy the hidden_states to save gpu space
    encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(0, expanded_return_idx.to(device))

    model_kwargs = {"encoder_outputs": encoder_outputs,}
    eos, bos = tokenizer.eos_token_id, tokenizer.bos_token_id

    assert model.config.model_type == "bart"
    decoder_input_ids_base = torch.LongTensor([[model.config.decoder_start_token_id]]).to(device)

    # for original
    min_length = gen_target_min
    max_length = gen_target_max
    logits_processor = get_logits_processor(model.config, encoder_input_ids = encoder_input_ids, min_length = min_length, max_length = max_length,num_beams = num_beams,)
    length_penalty = model.config.length_penalty # default is 2
    early_stopping = model.config.early_stopping # true
    num_return_sequences = model.config.num_return_sequences
    beam_scorer = BeamSearchScorer(batch_size=1,num_beams=num_beams,device=device,length_penalty=length_penalty,do_early_stopping=early_stopping,num_beam_hyps_to_keep=num_return_sequences,)
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(MaxLengthCriteria(max_length=max_length))

    with torch.no_grad():
        decoder_input_ids = decoder_input_ids_base.index_select(0, expanded_return_idx.to(device))

        # model.debug_count = 0 # DEBUG
        outputs = beam_search(
            model,
            decoder_input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=model.config.pad_token_id if model.config.pad_token_id is not None else eos,
            eos_token_id=eos,
            return_dict_in_generate=model.config.return_dict_in_generate,
            **model_kwargs,
        )
        if isinstance(outputs, BeamSearchEncoderDecoderOutput) or isinstance(outputs, BeamSearchDecoderOnlyOutput):
            outputs = outputs.sequences
        assert outputs.dim() == 2 and outputs.size(0) == 1

    text = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
    loss = None
    return text

class DynamicModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class SummarizationModel(DynamicModel):
    model_name_or_path: str 
    output_dir: str
    max_source_length: int = 2048
    gen_target_max: int = 400
    gen_target_min: int = 20
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    grad_accumulation: int = 1
    seed: int = 42
    do_pretrain: bool = False
    warmup_ratio: float = 0
    learning_rate: float = 5e-5
    epochs_pretrain: int = 3
    epochs_finetune: int = 3
    save_total_limit: int = 1
    eval_with_generate: bool = False
    eval_steps:int=500
    save_steps:int=500
    # resume_from_checkpoint: str

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def get_epochs(self) -> int:
        return self.epochs_pretrain if self.do_pretrain else self.epochs_finetune

    def get_train_args(self, do_eval: bool) -> Seq2SeqTrainingArguments:
        return Seq2SeqTrainingArguments(
            seed=self.seed,
            do_train=True,
            do_eval=do_eval or None,  # False still becomes True after parsing
            overwrite_output_dir=True,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size = self.per_device_eval_batch_size, #  SCH: limit if need to eval rouge
            gradient_accumulation_steps=self.grad_accumulation,
            warmup_ratio=self.warmup_ratio,
            output_dir=self.output_dir,
            save_total_limit=self.save_total_limit,
            # save_strategy=IntervalStrategy.EPOCH,
            save_strategy="steps",
            # evaluation_strategy=IntervalStrategy.EPOCH
            evaluation_strategy="steps"
            if do_eval
            else IntervalStrategy.NO,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", 
            greater_is_better=False,  # Set to true if using Rouge, False if using loss
            learning_rate=self.learning_rate,
            num_train_epochs=self.get_epochs(),
            # predict_with_generate=self.predict_with_generate, # Enable this to enable the calculation of rouge during validation
            predict_with_generate=self.eval_with_generate, # Enable this to enable the calculation of rouge during validation
            # resume_from_checkpoint=self.resume_from_checkpoint,
            # log_level="passive",
        )

    # def fit(self, train_file, validation_file):
    def fit(self, train_file, validation_file):
        """
        train with or without eval by setting --do_eval
        """
        train_script = run_summarization
        data_args = train_script.DataTrainingArguments(
            train_file=train_file,
            validation_file=validation_file,
            overwrite_cache=True,
            # max_target_length=self.gen_target_max,
            max_source_length=self.max_source_length,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            text_column="text",
            summary_column="summary",
        )
        train_args = self.get_train_args(args.do_eval)
        model_args = train_script.ModelArguments(
            model_name_or_path=self.model_name_or_path,
            gen_target_max=self.gen_target_max,
            gen_target_min=self.gen_target_min,
        )
        train_script.main(
            model_args=model_args, data_args=data_args, training_args=train_args, \
        )


def main(args):  

    data_dir = str(Path(args.output_dir) / "data")


    # Training
    if args.do_train:
        model = SummarizationModel(
            model_name_or_path=args.model_name_or_path,
            output_dir=args.output_dir,
            max_source_length=args.max_source_length,
            gen_target_max=args.gen_target_max,
            gen_target_min=args.gen_target_min,
            seed=args.seed,
            do_pretrain=False,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.lr,
            save_total_limit=args.save_total_limit,
            eval_with_generate=args.eval_with_generate,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            # resume_from_checkpoint=args.resume_from_checkpoint,
        )
        model.fit(args.train_file, args.validation_file)

    # Generation
    if args.do_predict:
        config = AutoConfig.from_pretrained(
            args.output_dir if args.do_train else args.model_name_or_path
        )
        config.gen_target_max = args.gen_target_max
        config.gen_target_min = args.gen_target_min
        config.return_dict_in_generate = args.return_dict_in_generate
        config.length_penalty = args.length_penalty
        config.early_stopping = args.early_stopping
        # SentBS: allow source length > 1024
        if args.max_source_length > 1024:
            print("setting max position embedding:", args.max_source_length)
            config.max_position_embeddings = args.max_source_length

        print("using configuration:\n", config)
        lm = AutoModelForSeq2SeqLM.from_pretrained(
            args.output_dir if args.do_train else args.model_name_or_path, 
            # return_dict_in_generate=args.return_dict_in_generate, 
            config=config,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir if args.do_train else args.model_name_or_path, 
            use_fast=True, # use fast tokenizer
        ) 
        lm.resize_token_embeddings(len(tokenizer))
        
        df_test = pd.read_csv(args.test_file)
        df_test = df_test[['text', 'summary']]
        text_list = df_test["text"].tolist()
        target_list = df_test["summary"].tolist()
            

        raw_golds,raw_preds = [],[]
        if not os.path.exists(Path(args.output_dir)):
            os.mkdir(Path(args.output_dir))

        raw_prediction_file = open(str(Path(args.output_dir) / "raw_prediction.txt"),'w',encoding='utf-8')
        # raw_results_file = open(str(Path(args.output_dir) / "raw_rouge_results.json"),'w',encoding='utf-8')
        num_prediction_examples = len(text_list) if args.num_predict == -1 else min(args.num_predict, len(text_list))
        
        
        for i in tqdm(range(num_prediction_examples)):
            x, y = text_list[i], target_list[i]
            output = generate_beam_search(x, lm, tokenizer, max_source_length=args.max_source_length, num_beams=args.num_beams, gen_target_min=args.gen_target_min, gen_target_max=args.gen_target_max)
            raw_prediction_file.write(output.strip()+'\n')
            raw_preds.append(output)
            raw_golds.append(y)
            
                
def parse_arguments(parser):

    parser.add_argument('--seed', type=int, default=42, help="random seed")

    parser.add_argument('--do_train', action="store_true", default=False,
                        help="train model")
    parser.add_argument('--do_eval', action="store_true", default=False,
                        help="eval model")
    parser.add_argument('--do_predict', action="store_true", default=False,
                        help="use model for inference")

    parser.add_argument('--model_name_or_path', type=str, default="",
                        help="model name or path")
                                            
    parser.add_argument('--train_file', type=str, default="",
                        help="train file path")
    parser.add_argument('--validation_file', type=str, default="",
                        help="validation file path")
    parser.add_argument('--test_file', type=str, default="",
                        help="test file path")

    parser.add_argument('--output_dir', type=str, default="result/default/",
                        help="locaton to store model files")
    parser.add_argument('--save_total_limit', default=False, help="maximum number of models stored in the output dir")

    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="deviec batch size for training")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="deviec batch size for validation")

    parser.add_argument('--max_train_samples', type=int, default=None, help="deviec batch size for training")
    parser.add_argument('--max_eval_samples', type=int, default=None, help="deviec batch size for validation")

    parser.add_argument('--gen_target_min', type=int, default=20, help="model minimun number of tokens for target")
    parser.add_argument('--gen_target_max', type=int, default=400, help="model maximum number of tokens for target")

    parser.add_argument('--length_penalty', type=float, default=2.0, help="length penalty for beam scorer")


    parser.add_argument('--max_source_length', type=int, default=2048, help="maximum number of tokens for source")

    parser.add_argument('--num_beams', type=int, default=4, help="number of beams used for beam search")
    
    parser.add_argument('--num_predict', type=int, default=-1, help="number of test examples to run for prediction")

    parser.add_argument('--eval_steps', type=int, default=500, help="number of training steps to run evaluation")

    parser.add_argument('--early_stopping', action="store_true", default=False, help="early stopping during generation")

    # deprecated, use eval_with_generate instead
    parser.add_argument('--predict_with_generate', action="store_true", default=False, help="Whether to use generate to calculate generative metrics (ROUGE, BLEU).")
    
    parser.add_argument('--eval_with_generate', action="store_true", default=False, help="Whether to use generate to calculate generative metrics (ROUGE, BLEU).")

    parser.add_argument('--return_dict_in_generate', type=bool, default=False, help="whether to return additional informations during genration")

    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ctrl transformer implementation")
    args = parse_arguments(parser)
    main(args)


