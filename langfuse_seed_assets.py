"""Gera e publica prompt + dataset de classificacao no Langfuse."""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langfuse import get_client, observe
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEFAULT_PROMPT_NAME = "classificador-compras-publicas"
DEFAULT_PROMPT_LABEL = "production"
DEFAULT_DATASET_NAME = "compras-publicas-classificacao"
DEFAULT_PROMPT_MODEL = "mistral-small-latest"
DEFAULT_ITEMS_JSON_PATH = "raw_items_seed.json"
TRAIN_SPLIT_RATIO = 0.8
SPLIT_RANDOM_SEED = 42
MAX_MISTRAL_WORKERS = 5

PROMPT_TEMPLATE = """Você é um classificador especializado em itens de compras públicas brasileiras.
Dado o item abaixo, classifique como SERVIÇO ou PRODUTO, informe confiança e justificativa breve.

Item: {{item_descricao}}

Responda sempre neste formato exato:
CLASSIFICAÇÃO: [SERVIÇO ou PRODUTO]
CONFIANÇA: [Alta, Média ou Baixa]
JUSTIFICATIVA: [1-2 frases]
"""


def _safe_get(obj: Any, field: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


class SeedRunner:
    def __init__(self) -> None:
        self.prompt_name = DEFAULT_PROMPT_NAME
        self.prompt_label = DEFAULT_PROMPT_LABEL
        self.prompt_model = DEFAULT_PROMPT_MODEL
        self.dataset_name = DEFAULT_DATASET_NAME
        self.items_json_path = os.getenv("LANGFUSE_ITEMS_JSON_PATH", DEFAULT_ITEMS_JSON_PATH)
        self.langfuse = get_client()

    @observe(name="seed.classify_with_mistral", as_type="generation")
    def classify_with_mistral(self, descricao: str) -> Dict[str, str]:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY não definida. Sem fallback, a classificação não pode continuar.")

        from mistralai import Mistral

        client = Mistral(api_key=api_key)
        system_prompt = """
        Você é um especialista em compras públicas.
        Retorne APENAS JSON válido com chaves:
        - classificacao: SERVIÇO ou PRODUTO
        - confianca: Alta, Média ou Baixa
        - justificativa: texto curto
        """
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classifique este item:\n\n{descricao}"},
            ],
            max_tokens=180,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").replace("json", "", 1).strip()

        payload = json.loads(content)
        return {
            "classificacao": payload.get("classificacao", "PENDENTE"),
            "confianca": payload.get("confianca", "Média"),
            "justificativa": payload.get("justificativa", "Classificação gerada automaticamente."),
        }

    @observe(name="seed.generate_single_item")
    def generate_single_item(self, descricao: str) -> Dict[str, Any]:
        result = self.classify_with_mistral(descricao)
        return {
            "input": {"item_descricao": descricao},
            "expected_output": {
                "classificacao": result["classificacao"],
                "confianca": result["confianca"],
                "justificativa": result["justificativa"],
            },
        }

    @observe(name="seed.generate_items")
    def generate_items(self, raw_items: List[str]) -> List[Dict[str, Any]]:
        if not raw_items:
            return []

        total = len(raw_items)
        workers = min(MAX_MISTRAL_WORKERS, total)
        items: List[Optional[Dict[str, Any]]] = [None] * total

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(self.generate_single_item, descricao): idx
                for idx, descricao in enumerate(raw_items)
            }

            for future in tqdm(
                as_completed(future_to_index),
                total=total,
                desc=f"Gerando itens com Mistral ({workers} workers)",
                unit="item",
            ):
                idx = future_to_index[future]
                items[idx] = future.result()

        if any(item is None for item in items):
            raise RuntimeError("Falha ao gerar todos os itens do seed.")

        return [item for item in items if item is not None]

    def ensure_prompt(self) -> None:
        try:
            self.langfuse.create_prompt(
                name=self.prompt_name,
                prompt=PROMPT_TEMPLATE,
                labels=[self.prompt_label],
                tags=["compras-publicas", "classificacao"],
                type="text",
                config={"model": self.prompt_model},
            )
            print(f"Prompt criado: {self.prompt_name}@{self.prompt_label} (model={self.prompt_model})")
        except Exception as error:
            print(f"Prompt ja existente ou erro nao bloqueante: {error}")

    def ensure_dataset(self) -> Optional[str]:
        dataset_id: Optional[str] = None
        try:
            dataset = self.langfuse.create_dataset(
                name=self.dataset_name,
                description="Dataset de classificacao de itens de compras publicas",
                metadata={"origem": "langfuse_seed_assets.py"},
            )
            dataset_id = _safe_get(dataset, "id")
            print(f"Dataset criado: {self.dataset_name}")
        except Exception as error:
            print(f"Dataset ja existente ou erro nao bloqueante: {error}")
            try:
                dataset = self.langfuse.get_dataset(self.dataset_name)
                dataset_id = _safe_get(dataset, "id")
            except Exception:
                dataset_id = None
        return dataset_id

    def create_dataset_item(
        self,
        *,
        item_input: Dict[str, Any],
        item_expected_output: Dict[str, Any],
        item_metadata: Dict[str, Any],
    ) -> None:
        self.langfuse.create_dataset_item(
            dataset_name=self.dataset_name,
            input=item_input,
            expected_output=item_expected_output,
            metadata=item_metadata,
        )

    @staticmethod
    def load_raw_items(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, list) and payload and isinstance(payload[0], str):
            return payload

        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            output: List[str] = []
            for item in payload:
                if "item_descricao" in item:
                    output.append(str(item["item_descricao"]))
            if output:
                return output

        raise ValueError("Arquivo JSON deve ser lista de strings ou objetos com item_descricao.")

    @staticmethod
    def build_train_test_splits(total_items: int) -> List[str]:
        indices = list(range(total_items))
        random.Random(SPLIT_RANDOM_SEED).shuffle(indices)
        train_count = int(total_items * TRAIN_SPLIT_RATIO)
        train_indices = set(indices[:train_count])
        return ["train" if idx in train_indices else "test" for idx in range(total_items)]

    @observe(name="seed.run")
    def run(self) -> None:
        raw_items = self.load_raw_items(self.items_json_path)
        generated_items = self.generate_items(raw_items)
        print(f"Itens gerados: {len(generated_items)}")

        # Sem dependência entre si: cria/garante prompt e dataset em paralelo.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_prompt = executor.submit(self.ensure_prompt)
            future_dataset = executor.submit(self.ensure_dataset)
            future_prompt.result()
            future_dataset.result()

        split_flags = self.build_train_test_splits(len(generated_items))
        train_total = split_flags.count("train")
        test_total = split_flags.count("test")
        print(f"Split definido: train={train_total} | test={test_total}")

        sent = 0
        for idx, item in enumerate(generated_items):
            split = split_flags[idx]
            self.create_dataset_item(
                item_input=item["input"],
                item_expected_output=item["expected_output"],
                item_metadata={"split": split},
            )
            sent += 1

        self.langfuse.flush()
        print(f"Itens enviados ao dataset: {sent}/{len(generated_items)}")


if __name__ == "__main__":
    SeedRunner().run()
