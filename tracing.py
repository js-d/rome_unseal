import os
from pathlib import Path

from typing import Union, Tuple, Optional
from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from tqdm import tqdm
import unseal
from unseal.hooks import Hook, HookedModel, common_hooks
from unseal.transformers_util import load_from_pretrained, get_num_layers





def prepare_input(
    text: str,
    tokenizer: transformers.AutoTokenizer,
    device: Optional[Union[str, torch.device]] = "cpu",
) -> Tuple[dict, int]:
    encoded_text = tokenizer(text, return_tensors="pt").to(device)

    # split correct target token from sentence
    correct_id = encoded_text["input_ids"][0, -1].item()
    encoded_text["input_ids"] = encoded_text["input_ids"][:, :-1]
    encoded_text["attention_mask"] = encoded_text["attention_mask"][:, :-1]

    return encoded_text, correct_id


def get_noise_hook():
    # unseal has a hook to add gaussian noise on the output of any module
    mean = 0
    std = 0.1
    embedding_name = "transformer->wte"  # name of the embedding module
    entity_indices = "0:4"  # add noise to the first 4 token embeddings, which are the subject in our case
    noise_hook = Hook(
        layer_name=embedding_name,
        func=unseal.hooks.common_hooks.additive_output_noise( # TODO: had to change rome_hooks to common_hooks here (rome_hooks is deprecated)
            indices=f"{entity_indices},:", mean=mean, std=std
        ),
        key="embedding_noise",
    )
    return noise_hook


def get_patch_hook(clean_outputs, layer, position):
    # function that inserts the stored 'clean' hidden states into the model at the appropriate position
    # uses Unseal's 'hidden_patch_hook_fn' for this, which is just a thin wrapper around the replace_activation hook
    return Hook(
        layer_name=f"transformer->h->{layer}",
        func=unseal.hooks.common_hooks.hidden_patch_hook_fn(
            position, clean_outputs[layer][0, position]
        ),
        key=f"patch_{layer}_pos{position}",
    )


def compute_trace(model, hidden_states, encoded_text, correct_id, noise_hook, num_layers, num_tokens):
    print("Patching hidden states...")
    results = torch.empty((num_tokens, num_layers))
    for layer in tqdm(range(num_layers)):
        for position in range(num_tokens):
            hook = get_patch_hook(hidden_states, layer, position)

            output = model(**encoded_text, hooks=[noise_hook] + [hook])

            prob = torch.softmax(output["logits"][0, -1, :], 0)[correct_id].item()
            results[position, layer] = prob
    return results


def save_trace_plot(results, save_path):
    num_tokens, num_layers = results.shape[0], results.shape[1]
    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.pcolormesh(results, cmap="Purples")
    # TODO: change the labelling here
    ax.set_yticks(
        np.arange(num_tokens) + 0.5, ["The", "Big", "Bang", "Theory", "premie", "res", "on"]
    )
    ax.set_xticks(np.arange(0, num_layers, 5) + 0.5, np.arange(0, num_layers, 5))
    cbar = plt.colorbar(image)
    fig.gca().invert_yaxis()
    cbar.ax.set_title("p(CBS)", y=-0.07)
    fig.savefig(save_path)


def main(model, text, path):
    

    # prepare text input
    encoded_text, correct_id = prepare_input(text, tokenizer, device)
    num_tokens = encoded_text["input_ids"].shape[1]

    # prediction in the uncorrupted run:
    print(
        f"Uncorrupted performance: {model(**encoded_text, hooks=[])['logits'].softmax(dim=-1)[0,-1,correct_id].item():.5f}"
    )

    # hidden states in the uncorrupted run
    num_layers = get_num_layers(model, layer_key_prefix="transformer->h")
    output_hooks_names = [f"transformer->h->{layer}" for layer in range(num_layers)]
    output_hooks = [
        Hook(name, common_hooks.save_output(False, False), name) for name in output_hooks_names # TODO: had to add the False, False to make this work
    ]
    model(**encoded_text, hooks=output_hooks)
    hidden_states = [
        model.save_ctx[f"transformer->h->{layer}"]["output"][0]
        for layer in range(num_layers)
    ]

    # implement the corruption noise
    noise_hook = get_noise_hook()

    # prediction in the corrupted run
    print(
        f"Corrupted performance: {model(**encoded_text, hooks=[noise_hook])['logits'].softmax(dim=-1)[0,-1,correct_id]:.5f}"
    )
    # A drop of 97%! (Note that this can vary a lot due to the variance of the noise)

    # get trace
    results = compute_trace(model, hidden_states, encoded_text, correct_id, noise_hook, num_layers, num_tokens)

    # save figure
    save_trace_plot(results, path)


if __name__ == "__main__":
    model_name = "gpt2-large"
    text_list = ["The Big Bang Theory premieres on CBS"]
    
    # let's load the model
    unhooked_model, tokenizer, config = load_from_pretrained(model_name)

    # hook up the model
    model = HookedModel(unhooked_model)

    # put it on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Model loaded and on device {device}!")

    # create a path to save traces for this model
    model_path = Path("traces", model_name)
    os.makedirs(model_path, exist_ok=True)

    # get trace for every text in text list
    for (text_idx, text) in enumerate(text_list):
        path = model_path / Path(f"{text_idx}.png")
        main(model, text, path)