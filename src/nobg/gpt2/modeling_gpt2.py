import torch
from torch import nn
from typing import Any, Optional, Union
from dataclasses import dataclass

from transformers import GPT2Config as _TransformersGPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from ..mixin import Revised_Mixin
from ..utils import model_card_template


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12


class GPT2(
    nn.Module,
    Revised_Mixin,
    library_name="nobg",
    repo_url="https://github.com/feyninc/nobg",
    tags=["text-generation", "nobg", "gpt2"],
    model_card_template=model_card_template(
        class_name="GPT2", default_repo="nobg/gpt2"
    ),
):
    """an AI model for visual question answering"""

    def __init__(
        self,
        config: Optional[GPT2Config] = None,
    ):
        """
        Initialize the GPT2 model with the given parameters.

        Args:
            config (GPT2Config, optional): The configuration object for the model. If None, default values will be used.

        """

        super().__init__()
        self.config = config or GPT2Config()

        _tf_config = _TransformersGPT2Config(
            vocab_size=self.config.vocab_size,
            n_positions=self.config.n_positions,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
        )

        self.wte = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embedding(self.config.n_positions, self.config.n_embd)
        self.h = nn.ModuleList(
            [GPT2Block(_tf_config, layer_idx=i) for i in range(self.config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, ...], ...]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        This method computes the forward pass of the model.
        It takes as input a tensor of token indices (ids) and computes the logits for the next token.

        Args:
            input_ids (torch.Tensor): A tensor of shape (B, T) containing token indices. B is the batch size


        Returns:
            torch.Tensor: A tensor of shape (B, T, vocab_size) containing the logits for the next token.
        """
        _B, T = input_ids.size()
        assert T <= self.config.n_positions, (
            f"cannot forward sequence of length {T}, block_size is {self.config.n_positions}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.wpe(pos)
        tok_emb = self.wte(input_ids)
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return {"loss": loss, "logits": logits}
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 10,
        tokenizer: Any = None,
        return_generated_only: bool = False,
    ) -> Union[torch.Tensor, str]:
        """
        Generate text from the model.

        Args:
            input_ids: The input ids for the model. Shape: (batch_size, seq_len)
            attention_mask: The attention mask for the model. Shape: (batch_size, seq_len)
            max_new_tokens: The maximum number of new tokens to generate.
            tokenizer: The tokenizer to use for decoding the generated tokens.
            return_generated_only: Whether to return only the generated tokens.

        Returns:
            The generated tokens or decoded string if tokenizer is provided.
        """
        collect: list[int] = []
        for _ in range(max_new_tokens):
            output = self(input_ids=input_ids, attention_mask=attention_mask)
            output_id = int(torch.argmax(output[0, -1]).item())
            collect.append(output_id)
            if tokenizer and output_id == tokenizer.eos_token_id:
                break
            input_ids = torch.unsqueeze(
                torch.cat([input_ids[0], torch.tensor([output_id])]), dim=0
            )
            attention_mask = torch.ones_like(input_ids)
        if return_generated_only:
            if tokenizer is None:
                return torch.tensor(collect)
            else:
                return tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(collect)
                )
        if tokenizer is not None:
            return tokenizer.batch_decode(input_ids)[0]
        return input_ids
