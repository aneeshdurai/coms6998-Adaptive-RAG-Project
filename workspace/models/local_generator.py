import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from workspace.utils.logging_utils import get_logger

logger = get_logger("models.local_generator")

class LocalChat:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        lora_path: str | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device.startswith("cuda") else None,
            torch_dtype=torch_dtype,
        )
        if lora_path:
            logger.info(f"Loading LoRA adapters from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

    @torch.no_grad()
    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = temperature > 0
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the generated completion portion if possible
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()
