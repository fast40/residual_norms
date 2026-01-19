import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datasets import load_dataset


dataset = load_dataset('cais/mmlu', 'all')

def format_question(index):
    # question subject choices answer
    return dataset['validation']['question'][index] + '\n' + '\n'.join(f'{letter}) {answer}' for letter, answer in zip(['A', 'B', 'C', 'D'], dataset['validation']['choices'][index]))

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# inputs = tokenizer('Here is some test text.', return_tensors='pt')


def inspect_hidden_states(model, inputs, inspector_func):
    norms = []
    hooks = []

    for gpt_block in model.transformer.h:
        # print(gpt_block.named_modules)
        hooks.append(gpt_block.register_forward_hook(
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

QUESTION_INDEX = 0

inputs = tokenizer(format_question(QUESTION_INDEX), return_tensors='pt')

plt.title(f'Average Mean, L2 Norm, and Variance of Residual Streams for Question #{QUESTION_INDEX}')

x_values = [f'layer #{i}' for i in range(1, len(model.transformer.h) + 1)]

norm = inspect_hidden_states(model, inputs, torch.norm)
norm_line, = plt.plot(x_values, norm, marker='o', label='norms')

mean = inspect_hidden_states(model, inputs, torch.mean)
mean_line, = plt.plot(x_values, mean, marker='o', label='means')

var = inspect_hidden_states(model, inputs, torch.var)
var_line, = plt.plot(x_values, var, marker='o', label='var')

def on_key(event):
    global QUESTION_INDEX
    print('key')

    if event.key == 'right':
        QUESTION_INDEX += 1
    elif event.key == 'left':
        QUESTION_INDEX -= 1

    if event.key in ('right', 'left'):
        inputs = tokenizer(format_question(QUESTION_INDEX), return_tensors='pt')

        norm = inspect_hidden_states(model, inputs, torch.norm)
        norm_line.set_data(x_values, norm)

        mean = inspect_hidden_states(model, inputs, torch.mean)
        mean_line.set_data(x_values, mean)

        var = inspect_hidden_states(model, inputs, torch.var)
        var_line.set_data(x_values, var)

        plt.title(f'Average Mean, L2 Norm, and Variance of Residual Contributions for Question #{QUESTION_INDEX}')

        plt.gcf().canvas.draw_idle()

plt.gcf().canvas.mpl_connect('key_press_event', on_key)

plt.xlabel('Layer Number')
plt.ylabel('Mean, Variance, or L2 Norm')
plt.legend()
plt.ylim(0)
plt.show()
