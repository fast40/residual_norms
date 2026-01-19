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

# print(inspect.getsource(model.transformer.h[0].forward))
# print(model.transformer.h[0].named_modules)
# quit()

def inspect_hidden_states(model, inputs):
    # input to ln2 is hidden state after attn
    # outputs[0] of entire layer is hidden state after mlp
    # inputs[0] of entire layer is hidden state before mlp

    def layer_callback(module, inputs, outputs):  # entire layer
        global before_attn
        before_attn = inputs[0].flatten(end_dim=-2)

        global after_mlp
        after_mlp = outputs[0].flatten(end_dim=-2)

    def ln_2_callback(module, inputs, outputs):  # ln2
        global after_attn, before_mlp
        after_attn = before_mlp = inputs[0].flatten(end_dim=-2)

    layer_hook = model.transformer.h[-1].register_forward_hook(layer_callback)
    ln_2_hook = model.transformer.h[-1].ln_2.register_forward_hook(ln_2_callback)

    with torch.no_grad():
        model.eval()
        model(**inputs, output_hidden_states=True)

    layer_hook.remove()
    ln_2_hook.remove()

    return before_attn, after_attn, before_mlp, after_mlp


before_attn, after_attn, before_mlp, after_mlp = zip(*[inspect_hidden_states(model, tokenizer(format_question(i), return_tensors='pt')) for i in range(100)])
before_attn = torch.cat(before_attn)
after_attn = torch.cat(after_attn)
before_mlp = torch.cat(before_mlp)
after_mlp = torch.cat(after_mlp)

attn_contribution = after_attn - before_attn
mlp_contribution = after_mlp - before_mlp

attn_cosine = cosine_similarity(attn_contribution, before_attn).mean()
mlp_cosine = cosine_similarity(mlp_contribution, before_mlp).mean()

print(attn_cosine)
print(mlp_cosine)
