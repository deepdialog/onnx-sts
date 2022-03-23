"""
Infer
"""

import os
import json
from functools import lru_cache

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
import numpy as np
from transformers import AutoTokenizer


def create_model_for_provider(model_path: str, provider: str= 'CPUExecutionProvider') -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session


@lru_cache(maxsize=2000)
def encode(x):
    encoded_input = tokenizer([x], padding=True, truncation=True)
    sess_out = sess.run(['output'], {
        "input_ids": np.array(encoded_input['input_ids'], dtype=np.int64),
        "attention_mask": np.array(encoded_input['attention_mask'], dtype=np.int64)
    })
    out = np.sum(sess_out[0], 1) / sess_out[0].shape[1]
    return out.tolist()[0]

@lru_cache(maxsize=2000)
def sim(a, b):
    va = encode(a)
    vb = encode(b)
    va, vb = np.array(va), np.array(vb)
    s = np.dot(va, vb) / (np.sqrt(np.sum(va ** 2)) * np.sqrt(np.sum(vb ** 2)))
    s = float(s)
    return s


def sim_adjust(a, b):
    s = sim(a, b)
    s = np.clip(s + adjust[0], 0, 10) * adjust[1]
    return float(s)


CURERENT_DIR = os.path.realpath(os.path.dirname(__file__))
adjust = json.load(open(os.path.join(CURERENT_DIR, 'adjust.json')))
sess = create_model_for_provider(os.path.join(CURERENT_DIR, './stsq.onnx'))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(CURERENT_DIR, './paraphrase-multilingual-mpnet-base-v2'))


if __name__ == '__main__':
    print(encode('你好'))
    print(sim('你好', 'hello'))
    print(sim_adjust('你好', 'hello'))
