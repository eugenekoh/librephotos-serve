import json
import os
from types import SimpleNamespace

import torch
from loguru import logger
from oscar.modeling.modeling_bert import BertForPersonalityImageCaptioning
from transformers.pytorch_transformers import BertConfig, BertTokenizer

from pic.py_bottom_up import get_image_features
from pic.utils import restore_training_settings, CaptionTensorizer

ARGS_FILE = './pic/args.json'
with open(ARGS_FILE) as f:
    args = SimpleNamespace(**json.load(f))

args.device = 'cpu'
args.n_gpu = 0

# Load pretrained model and tokenizer
config_class, model_class, tokenizer_class = BertConfig, BertForPersonalityImageCaptioning, BertTokenizer
checkpoint = args.eval_model_dir
assert os.path.isdir(checkpoint)
config = config_class.from_pretrained(checkpoint)
config.output_hidden_states = args.output_hidden_states
tokenizer = tokenizer_class.from_pretrained(checkpoint)
model = model_class.from_pretrained(checkpoint, config=config)
model.to(args.device)

# inference and evaluation
args = restore_training_settings(args)

# setup tensorizer
tensorizer = CaptionTensorizer(tokenizer, args.max_img_seq_length,
                               args.max_seq_length, args.max_seq_a_length, args.mask_prob, args.max_masked_tokens)


def generate(img, personality):
    features, od_labels = get_image_features(img)
    logger.info("processing input")
    personality = personality.strip()
    example = tensorizer.tensorize_example("", features, text_b=od_labels,
                                           personality=personality)
    batch = [torch.unsqueeze(t, 0) for t in example]
    logger.info("starting inference")

    with torch.no_grad():
        model.eval()
        cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
            tokenizer.convert_tokens_to_ids([tokenizer.cls_token,
                                             tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token, '.']
                                            )

        inputs = {'is_decode': True,
                  'input_ids': batch[0], 'attention_mask': batch[1],
                  'token_type_ids': batch[2], 'img_feats': batch[3],
                  'masked_pos': batch[4],
                  'do_sample': False,
                  'bos_token_id': cls_token_id,
                  'pad_token_id': pad_token_id,
                  'eos_token_ids': [sep_token_id, pad_token_id],
                  'mask_token_id': mask_token_id,
                  # for adding od labels
                  'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

                  # hyperparameters of beam search
                  'max_length': 40, 
                  'num_beams': 5,
                  "temperature": args.temperature,
                  "top_k": 0,
                  "top_p": 1,
                  "repetition_penalty": 1.2,
                  "length_penalty": 1,
                  "num_return_sequences": 1,  # not supported
                  "num_keep_best": 1,
                  }

        # captions, logprobs
        outputs = model(**inputs)
        cap = outputs[0][0][0]
        conf = torch.exp(outputs[1])[0]

        decoded_cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
        result = {'caption': decoded_cap, 'confidence': conf.item(), 'personality' : personality}
        logger.info("finished inference " + str(result))

    return result
