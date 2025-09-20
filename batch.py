########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time, random, json, math, gc
from tqdm import tqdm
from torch.nn import functional as F
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.4b-20250905-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-1.5b-20250429-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-2.9b-20250519-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g0a-7.2b-20250829-ctx4096"

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')

from reference.rwkv7 import RWKV_x070
model = RWKV_x070(args)

from reference.utils import TRIE_TOKENIZER, sampler_simple_batch
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

########################################################################################################

# prompts = ["The apple can be", "The cat can be"]
# prompts = ["The apple can't be", "The cat can't be"]
prompts = ["The apple can be", "The cat can't be", "他们发现，这", "Q: 1+1=?\nA: 1+1=2."]
BATCH_SIZE = len(prompts)

state = model.generate_zero_state(BATCH_SIZE)
init_outs = model.forward_batch([tokenizer.encode(prompt) for prompt in prompts], state)

for n in range(BATCH_SIZE):
    print(prompts[n])
    init_out = init_outs[n]
    probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
    _, indices = torch.topk(probs, 5) # print top-5 possibilities
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(repr(token), f'[probability {token_prob:.2%}]')
    if n != BATCH_SIZE-1:
        print()

########################################################################################################

prompts = ["也许", "我看到", "他们发现", "我认为", "哈哈", "这是一个有趣的", "List of Emojis:"]
BATCH_SIZE = len(prompts)
# prompts = ["这是一个有趣的"] * BATCH_SIZE
# prompts = ["他们发现"] * BATCH_SIZE
# prompts = ["我看到"] * BATCH_SIZE

state = model.generate_zero_state(BATCH_SIZE)
out = model.forward_batch([tokenizer.encode(prompt) for prompt in prompts], state)

tokens = []
GENERATE_LENGTH = 10
for i in range(GENERATE_LENGTH):
    new_tokens = sampler_simple_batch(out, noise=0).tolist()
    tokens.append(new_tokens)
    out = model.forward_batch(new_tokens, state)

tokens = np.transpose(np.array(tokens), axes=(1,0,2)).squeeze(-1)

print('\n')
for n in range(BATCH_SIZE):
    print(prompts[n], end='')
    print(tokenizer.decode(tokens[n], utf8_errors="ignore"))
    print('#'*80)
