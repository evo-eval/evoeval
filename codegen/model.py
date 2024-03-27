import os
from abc import ABC, abstractmethod
from typing import List
from warnings import warn

# Communism
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/ColossalTitan/huggingface/")

import anthropic
import google.generativeai as genai
import openai
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from vllm import LLM, SamplingParams

from evoeval.util.api_request import (
    create_anthropic_config,
    create_chatgpt_config,
    create_gemini_config,
    create_palm_config,
    num_tokens_from_messages,
    request_anthropic_engine,
    request_chatgpt_engine,
    request_gemini_engine,
    request_palm_engine,
)

HUMANEVAL_EOS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
NON_CODE_EOS = ["<|endoftext|>", "\n```", "\n</s>", "<|endofmask|>"]
EOS = HUMANEVAL_EOS + NON_CODE_EOS


# Adopted from https://github.com/huggingface/transformers/pull/14897
class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length : -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        conversational: bool = False,
        body: bool = False,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.conversational = conversational
        self.body = body

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class VLlmDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        conversational: bool = False,
    ) -> None:
        super().__init__(name, batch_size, temperature, max_new_tokens, conversational)
        kwargs = {"tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", "1"))}

        if "CodeLlama" in name:
            kwargs["dtype"] = "bfloat16"
        elif "code-millenials" in name:
            kwargs["dtype"] = "float16"
        elif "uukuguy/speechless-code-mistral-7b-v1.0" == name:
            kwargs["dtype"] = "float16"
        elif "uukuguy/speechless-codellama-34b-v2.0" == name:
            kwargs["dtype"] = "float16"
        elif "CodeBooga" in name:
            kwargs["dtype"] = "float16"
        elif "WizardCoder" in name and "V1.1" in name:
            kwargs["dtype"] = "bfloat16"
        elif "WizardCoder" in name:
            kwargs["dtype"] = "float16"
        elif "deepseek" in name:
            kwargs["dtype"] = "bfloat16"
        elif "mixtral" in name.lower():
            kwargs["dtype"] = "bfloat16"
        elif "solar" in name:
            kwargs["dtype"] = "float16"
        elif "mistral" in name.lower():
            kwargs["dtype"] = "bfloat16"
        elif "phi" in name.lower():
            kwargs["dtype"] = "float16"
            kwargs["trust_remote_code"] = True
        elif "openchat" in name.lower():
            kwargs["dtype"] = "bfloat16"

        # reset the eos
        self.eos = []
        self.llm = LLM(model=name, max_model_len=2048, **kwargs)

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens
                + len(self.llm.get_tokenizer().encode(prompt, return_tensors="pt")[0]),
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]

        return gen_strs


class CodeLlamaInstructSmall(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = f"""[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
```python
{prompt}
```
[/INST]
```python
"""

        return VLlmDecoder.codegen(self, input, do_sample, num_samples)


class Alpaca(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes request.

### Instruction:
Create a Python script for this problem:
{prompt}

### Response:
```python
"""
        return VLlmDecoder.codegen(self, prompt, do_sample, num_samples)


class OpenChat(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = f"""GPT4 Correct User: Can you complete the following Python function?
```python
{prompt}
```
<|end_of_turn|>GPT4 Correct Assistant:
```python
"""
        return VLlmDecoder.codegen(self, input, do_sample, num_samples)


class WizardCoderDecoder(VLlmDecoder):
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{prompt}

### Response:"""

        batch_size = min(self.batch_size, num_samples)

        num_of_tokens = len(
            self.llm.get_tokenizer().encode(prompt, return_tensors="pt")[0]
        )

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=num_of_tokens + self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
            ),
            use_tqdm=False,
        )

        return [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]


class XwinCoder(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:

        prompt = f"""<system>: You are an AI coding assistant that helps people with programming. Write a response that appropriately completes the user's request.
<user>: Complete the following code for me and return a fully runable code.
```python
{prompt}
```
<AI>:
```python
"""
        return VLlmDecoder.codegen(self, prompt, do_sample, num_samples)


class HFTorchDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {
            "trust_remote_code": name
            in {
                "bigcode/santacoder",
                "Salesforce/codegen2-1B",
                "Salesforce/codegen2-3_7B",
                "Salesforce/codegen2-7B",
                "Salesforce/codegen2-16B",
                "deepseek-ai/deepseek-coder-6.7b-base",
                "deepseek-ai/deepseek-coder-33b-base",
                "stabilityai/stable-code-3b",
                "Qwen/Qwen-14B-Chat",
                "Qwen/Qwen-7B-Chat",
            }
        }

        if "codegen-" in name:  # use fp16 for codegen models
            kwargs["torch_dtype"] = torch.float16
        if "codegen2-" in name:  # avoid warning of trust remote code
            kwargs["revision"] = "main"
            if "16b" in name.lower():
                kwargs["device_map"] = "auto"
        if "starcoder2" in name:
            kwargs["device_map"] = "auto"
        if "starcoder" in name:
            kwargs["torch_dtype"] = torch.bfloat16
        if "CodeLlama" in name:
            if "34b" in name.lower() or "70b" in name.lower():
                kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
            self.skip_special_tokens = True
        if "CodeBooga" in name:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            self.skip_special_tokens = True
        if "Mistral-7B-codealpaca-lora" == name:
            kwargs["torch_dtype"] = torch.float16
            self.skip_special_tokens = True
        elif "Mistral" in name or "zephyr-7b-beta" in name:
            kwargs["torch_dtype"] = torch.bfloat16
        if "Mixtral" in name:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
        if "deepseek" in name:
            kwargs["torch_dtype"] = torch.bfloat16
            if "33b" in name.lower():
                kwargs["device_map"] = "auto"
            self.skip_special_tokens = True
        if "/phi" in name:
            kwargs["torch_dtype"] = torch.float16
            kwargs["trust_remote_code"] = True
            self.skip_special_tokens = True
        if "Qwen" in name:
            kwargs["torch_dtype"] = torch.bfloat16
            self.skip_special_tokens = True
            if "72B" in name:
                kwargs["device_map"] = "auto"
        if "Phind" in name:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
        if "gemma" in name:
            kwargs["torch_dtype"] = torch.bfloat16
        if "Magicoder" in name:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"

        print(f"{kwargs = }")

        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if name in {"StabilityAI/stablelm-base-alpha-7b"}:
            print("Switching to float16 ...")
            self.model = self.model.half()
            self.skip_special_tokens = True

        if "device_map" not in kwargs:
            self.model = self.model.to(self.device)

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    # could be multiple eos in outputs, better pick minimum one
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class CodeLlamaInstructLarge(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos = ["\n```"]

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = f"""'<s>Source: system

 You are a helpful and honest code assistant expert in Python. Please, provide all answers to programming questions in Python.
 <step> Source: user

 Provide a self-contained Python script that solves the following problem:
```python
{prompt}
```
 <step> Source: assistant

 Here is a Python script that solves the problem:
```python
"""

        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        max_new_tokens = self.max_new_tokens + len(
            self.tokenizer.encode(prompt, return_tensors="pt")[0]
        )

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    # could be multiple eos in outputs, better pick minimum one
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class QwenInstruct(HFTorchDecoder):

    generation_template = "Please implement the following Python function in a markdown style code block:\n\n```python\n{prompt}\n```\n"
    incorrect_code_template = "```python\n{incorrect_solution}\n```\n"
    feedback_template = "{feedback}"

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1
        content = self.generation_template.format(prompt=prompt)

        input_tokens = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        max_token = len(input_tokens[0]) + self.max_new_tokens

        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=max_token,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            top_k=50,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        return gen_strs


class DeepSeekInstruct(HFTorchDecoder):

    generation_template = "Please implement the following Python function in a markdown style code block:\n\n```python\n{prompt}\n```\n"
    incorrect_code_template = "```python\n{incorrect_solution}\n```\n"
    feedback_template = "{feedback}"

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1
        content = self.generation_template.format(prompt=prompt)

        input_tokens = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        # set instruction model to have more max_tokens TODO: for all models
        max_token = len(input_tokens[0]) + self.max_new_tokens

        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=max_token,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            top_k=50,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=32021,
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        return gen_strs
        # return [x.split("```python")[-1].split("```")[0] for x in gen_strs]


class MistralInstruct(DeepSeekInstruct):
    pass  # just use the same as DeepSeekInstruct


class MixtralSPMXInstruct(DeepSeekInstruct):
    pass  # just use the same as DeepSeekInstruct


class GemmaInstruct(QwenInstruct):
    pass  # just use the same as QwenInstruct


class MagicCoderInstruct(DeepSeekInstruct):

    generation_template = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\nWrite a solution to the following problem:\n```python\n{prompt}\n```\n\n@@ Response\n"""

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1
        content = self.generation_template.format(prompt=prompt)

        input_tokens = self.tokenizer.encode(content, return_tensors="pt").to(
            self.device
        )

        max_token = len(input_tokens[0]) + self.max_new_tokens

        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=max_token,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            top_k=50,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        return gen_strs


class AnthropicDecoder(DecoderBase):
    generation_template = (
        "Please complete the following code snippet.\n```\n{prompt}\n```"
    )

    def __init__(self, name: str, model_name: str = "gpt-3.5-turbo", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.model_name = model_name
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "dummy")
        )

    def _anthrophic_parse(self, ret, prompt, body=False):
        outputs = []
        for returns in ret.content:
            raw_o = returns.text
            outputs.append(raw_o)
        return outputs

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        assert batch_size <= 20, "Use larger batch size could blow up the memory!"

        message = self.generation_template.format(prompt=prompt.strip())

        # estimation
        num_tokens = num_tokens_from_messages(message, self.model_name)

        config = create_anthropic_config(
            message=message,
            max_tokens=num_tokens + self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.model_name,
        )
        ret = request_anthropic_engine(self.client, config)
        return self._anthrophic_parse(ret, prompt.strip(), body=self.body)


class PalmDecoder(DecoderBase):
    generation_template = (
        "Please complete the following code snippet.\n```\n{prompt}\n```"
    )

    def __init__(self, name: str, model_name: str = "palm", **kwargs) -> None:
        super().__init__(name, **kwargs)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "dummy"))
        self.model_name = model_name

    def _palm_parse(self, ret, prompt):
        outputs = []
        raw_o = ret.result
        outputs.append(raw_o)
        return outputs

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        assert batch_size <= 20, "Use larger batch size could blow up the memory!"

        message = self.generation_template.format(prompt=prompt.strip())

        # approximate ge
        num_tokens = num_tokens_from_messages(message, self.model_name)

        config = create_palm_config(
            message=message,
            max_tokens=num_tokens + self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.model_name,
        )
        ret = request_palm_engine(genai, config)
        # if "gpt-3.5" in self.model_name:
        return self._palm_parse(ret, prompt.strip())


class GeminiChatDecoder(DecoderBase):
    generation_template = (
        "Please complete the following code snippet.\n```\n{prompt}\n```"
    )

    def __init__(
        self, name: str, model_name: str = "models/gemini-pro", **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "dummy"))
        self.model = genai.GenerativeModel(self.model_name)

    @staticmethod
    def _find_gen_func_sig(prompt):
        func_sig = ""
        for x in prompt.splitlines():
            if x.startswith("def ") and x.endswith(":"):
                # always pick the last one, since there could pre-defined functions.
                func_sig = x
        return func_sig

    @staticmethod
    def _remove_eos(gen):
        min_index = 100000000
        for eos in EOS:
            if eos in gen:
                min_index = min(min_index, gen.index(eos))
        return gen[:min_index]

    def _gemini_parse(self, ret, prompt):
        outputs = []
        raw_o = ret.text
        outputs.append(raw_o)
        return outputs

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        assert batch_size <= 20, "Use larger batch size could blow up the memory!"

        message = self.generation_template.format(prompt=prompt.strip())

        # approximate ge
        num_tokens = num_tokens_from_messages(message, self.model_name)

        config = create_gemini_config(
            max_tokens=num_tokens + self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
        )
        ret = request_gemini_engine(self.model, message, config)
        # if "gpt-3.5" in self.model_name:
        return self._gemini_parse(ret, prompt.strip())


class OpenAIChatDecoder(DecoderBase):
    generation_template = (
        "Please complete the following code snippet.\n```\n{prompt}\n```"
    )

    def __init__(self, name: str, model_name: str = "gpt-3.5-turbo", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.model_name = model_name
        openai.api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        assert batch_size <= 20, "Use larger batch size could blow up the memory!"

        # construct prompt
        # if "gpt-3.5" in self.model_name: # chatgpt
        message = self.generation_template.format(prompt=prompt.strip())

        num_tokens = num_tokens_from_messages(message, self.model_name)

        config = create_chatgpt_config(
            message=message,
            max_tokens=num_tokens + self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.model_name,
        )
        ret = request_chatgpt_engine(config)
        outputs = []
        for returns in ret.choices:
            outputs.append(returns.message.content)
        return outputs


class StarCoder2(HFTorchDecoder):
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = prompt.strip()  # starcoder2 needs this, bad
        return HFTorchDecoder.codegen(self, prompt, do_sample, num_samples)


class StarCoderInfill(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        temperature = max(self.temperature, 1e-2)
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


def make_model(name: str, batch_size: int = 1, temperature: float = 0.8):
    if name == "claude-3":
        return AnthropicDecoder(
            batch_size=batch_size,
            name="claude",
            temperature=temperature,
            model_name="claude-3-opus-20240229",
            conversational=True,
        )
    elif name == "claude-3-haiku":  # cheaper model
        return AnthropicDecoder(
            batch_size=batch_size,
            name="claude",
            temperature=temperature,
            model_name="claude-3-haiku-20240307",
            conversational=True,
        )
    elif name == "claude-2":
        return AnthropicDecoder(
            batch_size=batch_size,
            name="claude",
            temperature=temperature,
            model_name="claude-2.1",
            conversational=True,
        )
    elif name == "gemini-pro":
        return GeminiChatDecoder(
            batch_size=batch_size,
            name="gemini-pro",
            temperature=temperature,
            model_name="models/gemini-pro",
            conversational=True,
        )
    elif name == "palm":
        return PalmDecoder(
            batch_size=batch_size,
            name="palm",
            temperature=temperature,
            model_name="models/text-bison-001",
            conversational=True,
        )
    elif name == "chatgpt":
        return OpenAIChatDecoder(
            batch_size=batch_size,
            name="ChatGPT",
            temperature=temperature,
            model_name="gpt-3.5-turbo",
            conversational=True,
        )
    elif name == "gpt-4-turbo":
        return OpenAIChatDecoder(
            batch_size=batch_size,
            name="GPT4",
            temperature=temperature,
            model_name="gpt-4-turbo-preview",
            conversational=True,
        )
    elif name in ["gpt-4", "gpt-4-1106-preview"]:
        return OpenAIChatDecoder(
            batch_size=batch_size,
            name="GPT4",
            temperature=temperature,
            model_name=name,
            conversational=True,
        )
    elif name.startswith("starcoder2"):
        import re

        pattern = re.compile(r"starcoder2-(\d+)b")
        matches = pattern.findall(name)
        nb = int(matches[0])
        assert float(nb) > 0
        return StarCoder2(
            batch_size=batch_size,
            name=f"bigcode/{name}",
            temperature=temperature,
        )
    elif name.startswith("starcoder"):
        return StarCoderInfill(
            batch_size=batch_size, name=f"bigcode/{name}", temperature=temperature
        )
    elif name.startswith("code-llama-"):
        import re

        pattern = re.compile(r"code-llama-(\d+\.?\d*)b(.*)")
        matches = pattern.findall(name)[0]
        nb = matches[0]
        assert float(nb) > 0

        if "instruct" in name:
            if float(nb) < 69:  # nice
                return CodeLlamaInstructSmall(
                    batch_size=batch_size,
                    name=f"codellama/CodeLlama-{nb}b-Instruct-hf",
                    temperature=temperature,
                )
            else:
                return CodeLlamaInstructLarge(
                    batch_size=batch_size,
                    name=f"codellama/CodeLlama-{nb}b-Instruct-hf",
                    temperature=temperature,
                )
        elif "python" in name:
            return HFTorchDecoder(
                batch_size=batch_size,
                name=f"codellama/CodeLlama-{nb}b-Python-hf",
                temperature=temperature,
            )
        else:
            return VLlmDecoder(
                batch_size=batch_size,
                name=f"codellama/CodeLlama-{nb}b-hf",
                temperature=temperature,
            )
    elif name.startswith("deepseek-coder"):
        import re

        # format deepseek-coder-{nb}b*
        pattern = re.compile(r"deepseek-coder-(\d+\.?\d*)b(.*)")
        matches = pattern.findall(name)[0]
        nb = matches[0]
        assert float(nb) > 0

        if "instruct" in name:
            return DeepSeekInstruct(
                batch_size=batch_size,
                name=f"deepseek-ai/{name}",
                temperature=temperature,
                conversational=True,
            )
        else:
            return HFTorchDecoder(
                batch_size=batch_size,
                name=f"deepseek-ai/deepseek-coder-{nb}b-base",
                temperature=temperature,
            )
    elif name == "magicoder-s-ds-6.7b":
        return MagicCoderInstruct(
            batch_size=batch_size,
            name=f"ise-uiuc/Magicoder-S-DS-6.7B",
            temperature=temperature,
            conversational=True,
        )
    elif name == "magicoder-s-cl-7b":
        return MagicCoderInstruct(
            batch_size=batch_size,
            name=f"ise-uiuc/Magicoder-S-CL-7B",
            temperature=temperature,
            conversational=True,
        )
    elif name.startswith("wizardcoder-34b"):
        return WizardCoderDecoder(
            batch_size=batch_size,
            name=f"WizardLM/WizardCoder-Python-34B-V1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name.startswith("wizardcoder-33b-1.1"):
        return WizardCoderDecoder(
            batch_size=batch_size,
            name=f"WizardLM/WizardCoder-33B-V1.1",
            temperature=temperature,
            conversational=True,
        )
    elif name == "phind-code-llama-34b-v2":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Phind/Phind-CodeLlama-34B-v2",
            temperature=temperature,
        )
    elif name.startswith("mistral-7b"):
        if "instruct" in name:
            if name.endswith("-v02"):
                return MistralInstruct(
                    batch_size=batch_size,
                    name="mistralai/Mistral-7B-Instruct-v0.2",
                    temperature=temperature,
                    conversational=True,
                )
            else:
                return MistralInstruct(
                    batch_size=batch_size,
                    name="mistralai/Mistral-7B-Instruct-v0.1",
                    temperature=temperature,
                    conversational=True,
                )
        else:
            return HFTorchDecoder(
                batch_size=batch_size,
                name="mistralai/Mistral-7B-v0.1",
                temperature=temperature,
            )
    elif name.startswith("mixtral-8x7b"):
        if "instruct" in name:
            return MixtralSPMXInstruct(
                batch_size=batch_size,
                name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=temperature,
                conversational=True,
            )
        else:
            return HFTorchDecoder(
                batch_size=batch_size,
                name="mistralai/Mixtral-8x7B-v0.1",
                temperature=temperature,
            )
    elif name == "stable-code-3b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="stabilityai/stable-code-3b",
            temperature=temperature,
        )
    elif name == "speechless-codellama-34b":
        return Alpaca(
            batch_size=batch_size,
            name="uukuguy/speechless-codellama-34b-v2.0",
            temperature=temperature,
        )
    elif name == "openchat":
        return OpenChat(
            batch_size=batch_size,
            name="openchat/openchat-3.5-0106",
            temperature=temperature,
        )
    elif name.startswith("code-millenials-34b"):
        return Alpaca(
            batch_size=batch_size,
            name="budecosystem/code-millenials-34b",
            temperature=temperature,
            conversational=True,
        )
    elif name == "phi-2":
        return VLlmDecoder(
            batch_size=batch_size,
            name="microsoft/phi-2",
            temperature=temperature,
        )
    elif name.startswith("qwen"):
        # format deepseek-coder-{nb}b*
        import re

        pattern = re.compile(r"qwen-(\d+\.?\d*)b(.*)")
        matches = pattern.findall(name)[0]
        nb = matches[0]
        assert float(nb) > 0

        if "1.5" in name:
            return QwenInstruct(
                batch_size=batch_size,
                name=f"Qwen/Qwen1.5-{nb}B-Chat",
                temperature=temperature,
                conversational=True,
            )
        else:
            return QwenInstruct(
                batch_size=batch_size,
                name=f"Qwen/Qwen-{nb}B-Chat",
                temperature=temperature,
                conversational=True,
            )
    elif name.startswith("xwincoder-34b"):
        return XwinCoder(
            batch_size=batch_size, name="Xwin-LM/XwinCoder-34B", temperature=temperature
        )
    elif name.startswith("gemma"):
        import re

        pattern = re.compile(r"gemma-(\d+\.?\d*)b(.*)")
        matches = pattern.findall(name)[0]
        nb = matches[0]
        assert float(nb) > 0
        if "instruct" in name:
            return GemmaInstruct(
                batch_size=batch_size,
                name=f"google/gemma-{nb}b-it",
                temperature=temperature,
                conversational=True,
            )
        else:
            return HFTorchDecoder(
                batch_size=batch_size,
                name=f"google/gemma-{nb}b",
                temperature=temperature,
            )

    raise ValueError(f"Invalid model name: {name}")
