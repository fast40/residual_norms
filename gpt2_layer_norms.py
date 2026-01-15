import functools

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

inputs = tokenizer('Here is some test text.', return_tensors='pt')


def inspect_hidden_states(model, inputs, inspector_func):
    norms = []
    hooks = []

    for gpt_block in model.transformer.h:
        print(gpt_block.named_modules)
        hooks.append(gpt_block.mlp.register_forward_hook(
            lambda module, inputs, outputs: norms.append(inspector_func(outputs[0], dim=-1).mean()))
        )

    with torch.no_grad():
        model(**inputs, output_hidden_states=True)  # norms of hidden states captured by hooks

    for hook in hooks:
        hook.remove()

    return norms


ax = plt.gca()

ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.grid(which='major')
plt.grid(which='minor', alpha=0.3)

# for i in range(4, 25 + 1):

x_values = [f'layer #{i}' for i in range(1, len(model.transformer.h) + 1)]

norms = inspect_hidden_states(model, inputs, torch.norm)
plt.plot(x_values, norms, marker='o', label='norms')

means = inspect_hidden_states(model, inputs, torch.mean)
plt.plot(x_values, means, marker='o', label='means')

means = inspect_hidden_states(model, inputs, torch.var)
plt.plot(x_values, means, marker='o', label='var')

plt.title('Average Mean, L2 Norm, and Variance of Residual Contributions')
plt.xlabel('Layer Number')
plt.ylabel('Mean, Variance, or L2 Norm')
plt.legend()
plt.ylim(0)
plt.show()
