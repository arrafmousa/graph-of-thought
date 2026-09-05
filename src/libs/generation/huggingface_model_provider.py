"""Hugging Face local model provider (frozen base LLM, FP16 on a T4).

Loads a causal LM from an explicit repo id + revision, keeps it frozen, and
samples independent chains. For every generated token it records the selected
hidden layer's state (detached to CPU) and the sampled-token log-probability.
Heavy dependencies (torch, transformers) are imported lazily so this module can
be imported on a CPU-only machine for architecture tests (AGENTS.md section 31).
"""
from __future__ import annotations

import math

from .chain_trace import ChainTrace
from .model_provider import ModelProvider
from .token_node import TokenNode

_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


class HuggingFaceModelProvider(ModelProvider):
    """Sample token-level traces with hidden states from a frozen HF causal LM."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        dtype: str,
        device: str,
        hidden_layer: int,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported: {sorted(_DTYPES)}")
        self._model_id = model_id
        self._model_revision = model_revision
        self._dtype = dtype
        self._device = device
        self._hidden_layer = hidden_layer
        self._model = None
        self._tokenizer = None

    def provides_hidden_states(self) -> bool:
        return True

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def generate(
        self,
        *,
        prompt: str,
        num_chains: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int,
        seed: int,
    ) -> list[ChainTrace]:
        import torch

        model, tokenizer = self._ensure_loaded()
        inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_length = int(inputs["input_ids"].shape[1])

        traces: list[ChainTrace] = []
        for start in range(0, num_chains, batch_size):
            count = min(batch_size, num_chains - start)
            torch.manual_seed(seed + start)
            batch_inputs = {
                key: value.repeat(count, 1) for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            traces.extend(
                self._decode_batch(
                    outputs=outputs,
                    tokenizer=tokenizer,
                    prompt_length=prompt_length,
                    chain_offset=start,
                    count=count,
                    torch=torch,
                )
            )
            del outputs, batch_inputs
            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
        return traces

    def _decode_batch(self, *, outputs, tokenizer, prompt_length, chain_offset, count, torch):
        sequences = outputs.sequences
        num_steps = len(outputs.scores)
        eos_id = tokenizer.eos_token_id
        results: list[ChainTrace] = []
        for row in range(count):
            trace = ChainTrace(chain_id=chain_offset + row)
            terminated = "max_tokens"
            for step in range(num_steps):
                token_id = int(sequences[row, prompt_length + step].item())
                logits = outputs.scores[step][row]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                logprob = float(log_probs[token_id].item())
                hidden = outputs.hidden_states[step][self._hidden_layer][row, -1, :]
                vector = hidden.detach().to("cpu", dtype=torch.float32).tolist()
                text = tokenizer.decode([token_id], skip_special_tokens=False)
                trace.tokens.append(
                    TokenNode(
                        token_index=step,
                        token_id=token_id,
                        text=text,
                        logprob=logprob if math.isfinite(logprob) else float("-inf"),
                        hidden=vector,
                    )
                )
                if eos_id is not None and token_id == eos_id:
                    terminated = "eos"
                    break
            trace.terminated_reason = terminated
            generated_ids = [t.token_id for t in trace.tokens]
            trace.completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(trace)
        return results

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model, self._tokenizer
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, revision=self._model_revision
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch_dtype = getattr(torch, self._dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            revision=self._model_revision,
            torch_dtype=torch_dtype,
            output_hidden_states=True,
        )
        model.to(self._device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer
