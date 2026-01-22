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
    h_attn = []
    h_attn_mlp = []

    def ln_1_callback(module, inputs, outputs):
        h.append(inputs[0].flatten(end_dim=-2))

    def ln_2_callback(module, inputs, outputs):
        h_attn.append(inputs[0].flatten(end_dim=-2))

    def layer_callback(module, inputs, outputs):
        h_attn_mlp.append(outputs[0].flatten(end_dim=-2))

    hooks = []

    for layer in model.transformer.h:
        hooks.append(layer.ln_1.register_forward_hook(ln_1_callback))
        hooks.append(layer.ln_2.register_forward_hook(ln_2_callback))
        hooks.append(layer.register_forward_hook(layer_callback))

    model.eval()

    with torch.no_grad():
        model(**inputs, output_hidden_states=True)

    for hook in hooks:
        hook.remove()

    return h, h_attn, h_attn_mlp


h, h_attn, h_attn_mlp = zip(*[inspect_hidden_states(model, tokenizer(format_question(i), return_tensors='pt')) for i in range(10)])
h = [torch.cat(layer) for layer in zip(*h)]
h_attn = [torch.cat(layer) for layer in zip(*h_attn)]
h_attn_mlp = [torch.cat(layer) for layer in zip(*h_attn_mlp)]

# h, h_attn, h_attn_mlp = inspect_hidden_states(model, tokenizer(format_question(0), return_tensors='pt'))

print(h_attn_mlp[0])
print(h[1])

# attn_cosine = cosine_similarity(h_attn[-1] - h[-1], h[-1]).mean()
# mlp_cosine = cosine_similarity(h_attn_mlp[-1] - h_attn[-1], h_attn[-1]).mean()
#
# print(attn_cosine)
# print(mlp_cosine)

import matplotlib.pyplot as plt

plt.bar(range(len(h)), [(layer_h_attn_mlp.norm(dim=-1) - layer_h.norm(dim=-1)).mean() - (layer_h_attn_mlp.norm(dim=-1) - layer_h.norm(dim=-1)).median() for layer_h_attn_mlp, layer_h in zip(h_attn_mlp, h)], label='magnitude_change', color='tab:blue')
plt.ylabel('magnitude_change', color='tab:blue')
plt.tick_params(axis='y', labelcolor='tab:blue')
plt.twinx()
plt.plot([torch.acos(cosine_similarity(layer_h_attn_mlp - layer_h, layer_h).mean()) * (180 / torch.pi) for layer_h_attn_mlp, layer_h in zip(h_attn_mlp, h)], label='cosine_similarity', color='tab:red')
plt.ylabel('cosine_similarity', color='tab:red')
plt.tick_params(axis='y', labelcolor='tab:red')
plt.gca().set_ylim(ymin=0)

plt.xticks(range(len(h)), [f'layer #{i}' for i in range(1, len(h) + 1)])
# plt.plot([layer.norm(dim=-1).mean() for layer in h_attn], label='h_attn')
# plt.plot([layer.norm(dim=-1).mean() for layer in h_attn_mlp], label='h_attn_mlp')

plt.show()
