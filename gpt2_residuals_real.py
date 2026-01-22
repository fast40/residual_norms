import inspect

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


dataset = load_dataset('cais/mmlu', 'all')

def format_question(index):
    # question subject choices answer
    return dataset['validation']['question'][index] + '\n' + '\n'.join(f'{letter}) {answer}' for letter, answer in zip(['A', 'B', 'C', 'D'], dataset['validation']['choices'][index]))

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# print(inspect.getsource(model.transformer.forward))
# # print(model.transformer.h[0].named_modules)
# quit()

def inspect_hidden_states(model, inputs):
    # gpt2 layer: h -> ln_1 -> attn -> h + attn -> ln_2 -> mlp -> h + attn + mlp

    h = []
    hooks = []

    def ln_1_callback(module, inputs, outputs):
        h.append(inputs[0].flatten(end_dim=-2))

    def ln_2_callback(module, inputs, outputs):
        h.append(inputs[0].flatten(end_dim=-2))

    def ln_f_callback(module, inputs, outputs):
        h.append(inputs[0].flatten(end_dim=-2))
        h.append(outputs[0].flatten(end_dim=-2))

    for layer in model.transformer.h:
        hooks.append(layer.ln_1.register_forward_hook(ln_1_callback))
        hooks.append(layer.ln_2.register_forward_hook(ln_2_callback))

    hooks.append(model.transformer.ln_f.register_forward_hook(ln_f_callback))

    model.eval()

    with torch.no_grad():
        model(**inputs, output_hidden_states=True)

    for hook in hooks:
        hook.remove()

    return h


h = [inspect_hidden_states(model, tokenizer(format_question(i), return_tensors='pt')) for i in range(10)]
h = [torch.cat(layer) for layer in zip(*h)]

import matplotlib.pyplot as plt


xticks = ['e']

for i in range(1, len(model.transformer.h) + 1):
    xticks.append(f'attn{i}')
    xticks.append(f'mlp{i}')

xticks.append('ln_f')

plt.plot([layer_h.norm(dim=-1).mean() for layer_h in h], label='average norm')
plt.legend()
plt.xticks(range(len(h)), xticks)
# plt.plot([layer.norm(dim=-1).mean() for layer in h_attn], label='h_attn')
# plt.plot([layer.norm(dim=-1).mean() for layer in h_attn_mlp], label='h_attn_mlp')
plt.grid()

plt.show()
