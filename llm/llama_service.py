import mlflow
import time
from llama_cpp import Llama


class LLMEngine:
    def __init__(self, model_path: str):
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            verbose=False
        )

    def build_prompt(self, system: str, messages: list[tuple[str, str]]) -> str:
        """
        Builds a ChatML prompt from a list of (role, message) pairs.
        Roles must be: system, user, assistant
        """
        prompt = f"<|system|>\n{system}\n"
        for role, msg in messages:
            prompt += f"<|{role}|>\n{msg}\n"
        prompt += "<|assistant|>\n"
        return prompt

    def generate_text(self, system_msg: str, user_msg: str, max_tokens: int = 256):
        prompt = self.build_prompt(system_msg, [("user", user_msg)])

        with mlflow.start_run():
            mlflow.log_param("model_path", self.model.model_path)
            mlflow.log_param("system_msg", system_msg)
            mlflow.log_param("user_msg", user_msg)
            mlflow.log_param("max_tokens", max_tokens)
            mlflow.log_param("temperature", 0.5)
            mlflow.log_param("top_p", 0.9)
            mlflow.log_param("repeat_penalty", 1.15)

            start_time = time.time()

            output = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                stop=["<|user|>", "<|system|>", "</s>"],
                temperature=0.5,
                top_p=0.9,
                repeat_penalty=1.15
            )
            latency = time.time() - start_time

            response = output["choices"][0]["text"].strip()

            mlflow.log_metric("latency", latency)
            mlflow.log_text(response, "generated_response.txt")

        return response