import json
from os.path import abspath, dirname, exists, join
import argparse
import logging
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import socket
import os, sys
import re
import logging
from functools import partial
from demo_utils import download_model_folder
import argparse
import subprocess as sp

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import get_eval_list_same_length, load_model, boolean_string, fix_state_dict_namespace

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


EOS_ID = 50256


def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent


### FROM HUGGING FACE REPO
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits

def generate_next_token(model, input_ids, position_ids=None, token_type_ids=None, prev=None, temperature=1, top_k=0, top_p=0, past=None):
    with torch.no_grad():
        if not past:
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids, past=past)
        else:
            hidden_states, past = model.transformer(prev, past=past)
        logits = model.lm_head(hidden_states)
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits.unsqueeze(0), dim=-1)
        prev = torch.multinomial(probs, num_samples=1)
        return prev, probs[0][prev], past

def generate_sequence(model, input_ids, position_ids=None, token_type_ids=None, temperature=1, top_k=0, top_p=0, length=20, past=None, device='cuda'):
    output = input_ids.new_zeros([input_ids.size(0),0])
    prev = input_ids
    for i in range(length):
        prev, probs, past = generate_next_token(model, input_ids, position_ids, token_type_ids, prev, temperature, top_k, top_p, past)
        output = torch.cat((output, prev), dim=1)
    return output

def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", '-c', type=str, default='')
    parser.add_argument("--fp16", type=boolean_string, default=False)
    parser.add_argument("--max_seq_length", type=int, default=128)
    
    parser.add_argument("--generation_length", type=int, default=20)
    parser.add_argument("--max_history", type=int, default=2)

    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #### load the GPT-2 model 
    config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))
    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = load_model(GPT2LMHeadModel(config), args.load_checkpoint, args, verbose=True)
    model.to(device)
    model.eval()

    history = []
    while True:
        raw_text = input("USR >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("USR >>> ")
        history.append(raw_text)
        context_tokens = sum([enc.encode(h) + [EOS_ID] for h in history],[]) #+ [EOS_ID]
        context_tokens = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(0, context_tokens.size(-1), dtype=torch.long, device=context_tokens.device)

        out = generate_sequence(model, context_tokens, position_ids=position_ids,
                                length=args.generation_length, temperature=args.temperature, 
                                top_k=args.top_k, top_p= args.top_p) 

        out = out.tolist()                        
        text = enc.decode(cut_seq_to_eos(out[0])).encode('ascii','ignore').decode('ascii')
        print("SYS >>> ", text)
        history.append(text)
        history = history[-(2*args.max_history+1):]

if __name__ == '__main__':

    PYTHON_EXE = 'python'
    MODEL_FOLDER = './models'
    DATA_FOLDER = './data'

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
    )
    logger = logging.getLogger(__name__)


    if os.path.exists(MODEL_FOLDER):
        print('Found existing ./models folder, skip creating a new one!')
        os.makedirs(MODEL_FOLDER, exist_ok=True)
    else:
        os.makedirs(MODEL_FOLDER)

    #########################################################################
    # Download Model
    #########################################################################
    logger.info('Downloading models...')
    download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)

    # model size:  could be one of 'small' (GPT2 with 117M), 'medium'(345M) or 'large' (1542M)
    # dataset: one of 'multiref' or 'dstc'
    # from_scratch: True : load model trained from scratch or False: load model trained from fine-tuning the GPT-2
    target_folder = download_model(model_size='medium', dataset='multiref', from_scratch=False)
    logger.info('Done!\n')
    
    run_model()

