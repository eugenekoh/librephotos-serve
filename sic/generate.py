import torch
from loguru import logger
from oscar.modeling.modeling_bert import BertForImageCaptioning

from pic.py_bottom_up import get_image_features
from pic.utils import init
from sic.utils import CaptionTensorizer

args, model, tokenizer = init(BertForImageCaptioning, './sic/args.json')
# setup tensorizer
tensorizer = CaptionTensorizer(tokenizer, args.max_img_seq_length,
                               args.max_seq_length, args.max_seq_a_length, args.mask_prob, args.max_masked_tokens)


def generate(img):
    features, od_labels = get_image_features(img)
    logger.info("processing input")
    example = tensorizer.tensorize_example("", features, text_b=od_labels)
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
                  'max_length': 20,
                  'num_beams': 5,
                  "temperature": args.temperature,
                  "top_k": 0,
                  "top_p": 1,
                  "repetition_penalty": 1,
                  "length_penalty": 1,
                  "num_return_sequences": 1,  # not supported
                  "num_keep_best": 1,
                  }

        # captions, logprobs
        outputs = model(**inputs)
        cap = outputs[0][0][0]
        conf = torch.exp(outputs[1])[0]

        decoded_cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
        result = {'caption': decoded_cap, 'confidence': conf.item()}
        logger.info("finished inference " + str(result))

    return result
