import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

inputs = tokenizer('Here is some test text.', return_tensors='pt')


def get_norms(model, inputs):
    norms = []
    hooks = []

    for gpt_block in model.transformer.h:
        hooks.append(gpt_block.register_forward_hook(
            lambda module, inputs, outputs: norms.append(outputs[0].norm(dim=-1).mean()))
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

for i in range(4, 25 + 1):
    norms = get_norms(model, inputs)
    plt.plot(norms, marker='o')


plt.ylim(0)
plt.show()
