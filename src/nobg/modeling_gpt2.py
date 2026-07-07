# Copyright (c) 2022 Andrej Karpathy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
import math
from torch.nn import functional as F
from typing import (
    Optional,
    Tuple,
    Union,
)

from .mixin import Revised_Mixin


model_card_template = """
---
{{ card_data }}
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration.

Library: [nobg]({{repo_url}})

## how to load
```
pip install nobg
```

use the AutoModel class 
```python
from nobg AutoModel
model = AutoModel.from_pretrained("{{ repo_id | default("nobg/gpt2", true) }}")
```
or you can use the model class directly
```python
from nobg import GPT2
model = GPT2.from_pretrained("{{ repo_id | default("nobg/gpt2", true) }}")
```

## Contributions
Any contributions are welcome at https://github.com/feyninc/nobg

<img src="https://usefeyn.com/feyn/feyn_mark.svg"/>

"""


# huggingface/transformers implementation:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py


class MLP(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.c_proj = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.LongTensor):
        x = F.gelu(self.c_fc(x), approximate="tanh")
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embed, block_size) -> None:
        super().__init__()
        self.n_embed = n_embed
        assert (
            n_embed % n_head == 0
        ), f"n_head = {n_head} is not divisable by n_embed = {n_embed}"
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embed = n_embed
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.LongTensor):
        B, T, C = x.size()  # batch_size, sequence length, embedding_dim (n_dim)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        # nh = number of heads, hs = head size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        attn = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        return out


class Block(nn.Module):
    def __init__(self, n_head, n_embed, block_size) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_head, n_embed, block_size)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(
    nn.Module,
    Revised_Mixin,
    library_name="nobg",
    repo_url="https://github.com/feyninc/nobg",
    tags=["text-generation", "nobg", "gpt2"],
    model_card_template=model_card_template,
):
    """an AI model for visual question answering"""

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50257,
        n_layer: int = 11,
        n_head: int = 12,
        n_embed: int = 768,
    ):
        """
        Initialize the GPT2 model with the given parameters.

        Args:
            block_size (int, optional): The size of the input sequence. Defaults to 1024.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 5025
            n_layer (int, optional): The number of layers in the transformer. Defaults to 11.
            n_head (int, optional): The number of attention heads in each layer. Defaults to 12.
            n_embed (int, optional): The size of the embedding layer. Defaults to 768.

        """

        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embed),
                wpe=nn.Embedding(block_size, n_embed),
                h=nn.ModuleList(
                    [Block(n_head, n_embed, block_size) for _ in range(n_layer)]
                ),
                ln_f=nn.LayerNorm(n_embed),
            )
        )
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
        This method computes the forward pass of the model.
        It takes as input a tensor of token indices (ids) and computes the logits for the next token.

        Args:
            input_ids (torch.Tensor): A tensor of shape (B, T) containing token indices. B is the batch size


        Returns:
            torch.Tensor: A tensor of shape (B, T, vocab_size) containing the logits for the next token.
        """
        B, T = input_ids.size()
        assert (
            T <= self.block_size
        ), f"cannot forward sequence of length {T}, block_size is {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)  # shape T
        pos_emb = self.transformer.wpe(pos)  # positional embedding of shape (T,n_embed)
        tok_emb = self.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        # taken from original implementation
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return {"loss": loss, "logits": logits}
        return logits

    @classmethod
    def from_origin(cls, repo_id: str):
        """
        Loads a pretrained model from original huggingface repo, adapts them to the current model and injects them
        This due to mismatch in some layers (we are using Linear instead of Conv1D in the attention layer and other naming mismatches)
        how to use :
        ```python
        >>> model = GPT2LMHeadModel.from_origin('openai-community/gpt2')
        ```

        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            print(
                "Please install transformers library to load model weights from huggingface hub"
            )
            return None
        model_hf = AutoModelForCausalLM.from_pretrained(repo_id)
        params = ["vocab_size", "n_layer", "n_head"]
        kwargs = {k: getattr(model_hf.config, k, None) for k in params}
        model = cls(**kwargs)
        # sanitize state_dict
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param
        sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=10,
        tokenizer=None,
        return_generated_only=False,
        **kwargs,
    ) -> Union[torch.Tensor, str]:
        """
        Generate text from the model.
        This method will generate text by repeatedly feeding the model's output back into itself.
        It will stop generating text when it hits the end of text token or when it has generated `num_tokens` tokens.
        Args:
            input_ids (torch.Tensor): The input ids for the model. Shape: (batch_size, seq_len)
            attention_mask (torch.Tensor): The attention mask for the model. Shape: (batch_size, seq_len)
            max_new_tokens (int): The maximum number of new tokens to generate. Default: 10
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for decoding the generated tokens. Default: None
            return_generated_only (bool): Whether to return only the generated tokens or the full output from the model. Default: False
            **kwargs: Additional keyword arguments to pass to the model.
        Returns:
            torch.Tensor: The generated tokens or the full output from the model, depending on the value of `return_generated_only` (torch.Tensor) or a string (str) if `tokenizer` is provided.
        """
        collect = []
        for _ in range(max_new_tokens):
            output = self(input_ids=input_ids, attention_mask=attention_mask)
            output_id = torch.argmax(output[0, -1]).item()
            collect.append(output_id)
            if tokenizer and output_id == tokenizer.eos_token_id:
                break
            input_ids = torch.unsqueeze(
                torch.cat([input_ids[0], torch.tensor([output_id])]), dim=0
            )
            attention_mask = torch.ones_like(input_ids)
        # strip the input from the generated tokens
        if return_generated_only:
            if tokenizer is None:
                return torch.tensor(collect)
            else:
                tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(collect)
                )
        if tokenizer is not None:
            return tokenizer.batch_decode(input_ids)[0]
        else:
            return input_ids
