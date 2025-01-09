import torch
import torch.nn as nn
from Model.model import GPTModel
import numpy as np

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Left: {left.shape} and Right: {right.shape} shapes do not match")
    return nn.Parameter(torch.tensor(right))

def load_weights(model: GPTModel, params):
    model.pos_embedding.weight = assign(model.pos_embedding.weight, params['wpe'])
    model.token_embedding.weight = assign(model.token_embedding.weight, params['wte'])

    for b in range(len(params['blocks'])):
        W_q, W_k, W_v = np.split(params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1)
        model.transformer_blocks[b].attn.W_key.weight = assign(model.transformer_blocks[b].attn.W_key.weight, W_k.T)
        model.transformer_blocks[b].attn.W_query.weight = assign(model.transformer_blocks[b].attn.W_query.weight, W_q.T)
        model.transformer_blocks[b].attn.W_value.weight = assign(model.transformer_blocks[b].attn.W_value.weight, W_v.T)

        B_q, B_k, B_v = np.split(params['blocks'][b]['attn']['c_attn']['b'], 3, axis=-1)
        model.transformer_blocks[b].attn.W_key.bias = assign(model.transformer_blocks[b].attn.W_key.bias, B_k)
        model.transformer_blocks[b].attn.W_query.bias = assign(model.transformer_blocks[b].attn.W_query.bias, B_q)
        model.transformer_blocks[b].attn.W_value.bias = assign(model.transformer_blocks[b].attn.W_value.bias, B_v)

        model.transformer_blocks[b].attn.out_proj.weight = assign(model.transformer_blocks[b].attn.out_proj.weight, params['blocks'][b]['attn']['c_proj']['w'].T)
        model.transformer_blocks[b].attn.out_proj.bias = assign(model.transformer_blocks[b].attn.out_proj.bias, params['blocks'][b]['attn']['c_proj']['b'])

        model.transformer_blocks[b].ff.layers[0].weight = assign(model.transformer_blocks[b].ff.layers[0].weight, params['blocks'][b]['mlp']['c_fc']['w'].T)
        model.transformer_blocks[b].ff.layers[0].bias = assign(model.transformer_blocks[b].ff.layers[0].bias, params['blocks'][b]['mlp']['c_fc']['b'])
        model.transformer_blocks[b].ff.layers[2].weight = assign(model.transformer_blocks[b].ff.layers[2].weight, params['blocks'][b]['mlp']['c_proj']['w'].T)
        model.transformer_blocks[b].ff.layers[2].bias = assign(model.transformer_blocks[b].ff.layers[2].bias, params['blocks'][b]['mlp']['c_proj']['b'])

        model.transformer_blocks[b].norm1.scale = assign(model.transformer_blocks[b].norm1.scale, params['blocks'][b]['ln_1']['g'])
        model.transformer_blocks[b].norm1.shift = assign(model.transformer_blocks[b].norm1.shift, params['blocks'][b]['ln_1']['b'])
        model.transformer_blocks[b].norm2.scale = assign(model.transformer_blocks[b].norm2.scale, params['blocks'][b]['ln_2']['g'])
        model.transformer_blocks[b].norm2.shift = assign(model.transformer_blocks[b].norm2.shift, params['blocks'][b]['ln_2']['b'])
    
    model.final_norm.scale = assign(model.final_norm.scale, params['g'])
    model.final_norm.shift = assign(model.final_norm.shift, params['b'])
    model.out_head.weight = assign(model.out_head.weight, params['wte'])



def load_weights_into_gpt(gpt: GPTModel, params):
    gpt.pos_embedding.weight = assign(gpt.pos_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_query.weight = assign(
            gpt.transformer_blocks[b].attn.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].attn.W_key.weight = assign(
            gpt.transformer_blocks[b].attn.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].attn.W_value.weight = assign(
            gpt.transformer_blocks[b].attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_query.bias = assign(
            gpt.transformer_blocks[b].attn.W_query.bias, q_b)
        gpt.transformer_blocks[b].attn.W_key.bias = assign(
            gpt.transformer_blocks[b].attn.W_key.bias, k_b)
        gpt.transformer_blocks[b].attn.W_value.bias = assign(
            gpt.transformer_blocks[b].attn.W_value.bias, v_b)

        gpt.transformer_blocks[b].attn.out_proj.weight = assign(
            gpt.transformer_blocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].attn.out_proj.bias = assign(
            gpt.transformer_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].ff.layers[0].weight = assign(
            gpt.transformer_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].ff.layers[0].bias = assign(
            gpt.transformer_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].ff.layers[2].weight = assign(
            gpt.transformer_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].ff.layers[2].bias = assign(
            gpt.transformer_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
