import asyncio
import json
import re

from tqdm import tqdm
import aiohttp
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import get_proxy_access_token


def load_models_from_cfg(cfg: dict) -> dict:
    shared_models = {}  # {model_name: LLM instance}
    model_instances = {}  # {role_name: LLMInference instance}

    for role_name, model_cfg in cfg["models"].items():
        use_proxy = model_cfg.get("use_proxy", False)
        model_parameters = model_cfg.get("model_parameters", {})
        model_name = model_parameters.get("model", None)

        if use_proxy:
            llm = None
        else:
            if model_name in shared_models:
                print(
                    f"[INFO] Модель '{model_name}' уже загружена. Используем повторно с предыдущими параметрами"
                )
                llm = shared_models[model_name]
            else:
                print(
                    f"[LOAD] Загружается модель '{model_name}' для роли '{role_name}'..."
                )
                llm = LLM(**model_parameters)
                shared_models[model_name] = llm

        model_instances[role_name] = LLMInference(cfg=model_cfg, llm=llm)

    return model_instances


class LLMInference:
    def __init__(self, cfg: dict, llm: LLM = None):
        self.sampling_params = SamplingParams(**cfg["sampling_params"])
        self.use_proxy = cfg.get("use_proxy", False)
        self.model_parameters = cfg.get("model_parameters")
        self.max_user_prompt_len = cfg.get("max_user_prompt_len", None)
        self.system_msg = cfg.get("system_msg", None)
        self.enable_thinking = cfg.get("enable_thinking", None)
        self.batch_size = cfg.get("batch_size", 16)

        self.tokenizer = self.load_tokenizer()
        if self.use_proxy:
            self.proxy_model_name = cfg["proxy"]["model"]
            self.proxy_url = cfg["proxy"]["url"]
            self.proxy_max_retries = cfg["proxy"].get("max_retries", 5)
            self.proxy_base_delay = cfg["proxy"].get("base_delay", 90)
            self.access_token_env_name = cfg["proxy"]["env_proxy_access_token_name"]
            self.llm = None
        else:
            self.llm = llm if llm is not None else self.load_model()

    def load_tokenizer(self) -> AutoTokenizer:
        model_name = self.model_parameters["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_model_len = self.model_parameters.get("max_model_len")
        if max_model_len:
            tokenizer.model_max_length = max_model_len
        return tokenizer

    def load_model(self) -> LLM:
        return LLM(**self.model_parameters)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def prepare_prompts(self, user_prompts: list) -> list:
        prompts = []
        for prompt in user_prompts:
            if self.max_user_prompt_len:
                token_count = self.count_tokens(prompt)
                if token_count > self.max_user_prompt_len:
                    print(
                        f"[WARNING] Text truncated from {token_count} to {self.max_user_prompt_len} tokens."
                    )
                    prompt = self.tokenizer.decode(
                        self.tokenizer.encode(prompt)[: self.max_user_prompt_len]
                    )

            messages = [{"role": "user", "content": prompt}]
            if self.system_msg:
                messages.append({"role": "system", "content": self.system_msg})
            prompts.append(messages)
        return prompts

    async def async_proxy_infer_batch(self, batch: list) -> list:
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._async_proxy_infer_single(session, message) for message in batch
            ]
            return await asyncio.gather(*tasks)

    async def _async_proxy_infer_single(self, session: aiohttp.ClientSession, message: str) -> str:
        messages = self.prepare_prompts([message])[0]
        data = {
            "model": self.proxy_model_name,
            "messages": messages,
            "temperature": self.sampling_params.temperature,
            "max_tokens": self.sampling_params.max_tokens,
        }
        if self.enable_thinking is not None:
            data["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_proxy_access_token(self.access_token_env_name)}",
        }

        for attempt in range(self.proxy_max_retries):
            try:
                async with session.post(
                    self.proxy_url, headers=headers, json=data
                ) as resp:
                    if resp.status == 200:
                        try:
                            result = await resp.json()
                            return result["choices"][0]["message"]["content"]
                        except (aiohttp.ContentTypeError, json.JSONDecodeError):
                            text = await resp.text()
                            raise RuntimeError(f"Expected JSON but got: {text}")
                    elif resp.status in [429, 503]:
                        wait_time = self.proxy_base_delay * (attempt + 1)
                        print(f"[RETRY] Status {resp.status}. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"Unexpected status code {resp.status}: {await resp.text()}"
                        )
            except Exception as e:
                wait_time = self.proxy_base_delay * (attempt + 1)
                print(f"[ERROR] Attempt {attempt + 1}: {e}. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError("Max retries exceeded for proxy inference.")

    def batch_infer(self, batch: list) -> list:
        prompts = self.prepare_prompts(batch)
        if self.enable_thinking is not None:
            texts = [
                self.tokenizer.apply_chat_template(
                    p,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                for p in prompts
            ]
        else:
            texts = [
                self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
        inputs = [{"prompt": t} for t in texts]
        return self.llm.generate(
            inputs, sampling_params=self.sampling_params, use_tqdm=False
        )

    def run(self, prompts: list) -> list:
        batches = [
            prompts[i:i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        results = []

        for batch in tqdm(batches):
            print(len(batch))
            if self.use_proxy:
                result = asyncio.run(self.async_proxy_infer_batch(batch))
            else:
                outputs = self.batch_infer(batch)
                result = [out.outputs[0].text for out in outputs]
            results.extend(result)

        return results

    @staticmethod
    def delete_thinking_output(response: str) -> str:
        response = response.strip()
        match = re.search(r"</think>(.*)", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else response

    @staticmethod
    def parse_llm_comparison_output(response: str) -> str:
        cleaned = LLMInference.delete_thinking_output(response)
        matches = re.findall(r'"?(?:ОТВЕТ)"?:\s*([^\n]+)', cleaned, re.IGNORECASE)
        return "true" if matches and "true" in matches[-1].strip().lower() else "false"

    @staticmethod
    def parse_llm_response(response: str, fields: list) -> dict:
        cleaned = LLMInference.delete_thinking_output(response)
        data = {}
        for field in fields:
            match = re.search(rf'"?{field}"?\s*:\s*([^\n]+)', cleaned, re.IGNORECASE)
            val = match.group(1).strip().replace('"', "") if match else ""
            data[field] = "" if val.lower() == "не найдено" else val
        return data
