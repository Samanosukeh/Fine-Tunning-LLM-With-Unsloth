# Classificador de Compras Públicas — Fine-tuning com Unsloth

Fine-tuning de LLM para classificar itens de licitações públicas brasileiras como **SERVIÇO** ou **PRODUTO**, com nível de confiança e justificativa.

## Visão Geral

O projeto implementa um pipeline completo: geração de dataset via API, fine-tuning com LoRA e exportação do modelo — tudo rodando na GPU T4 gratuita do Google Colab.

```
Itens brutos → Rotulagem (Mistral API) → Fine-tuning (LoRA) → Classificador
```

**Exemplo de saída:**

```
INPUT:  Contratação de empresa para realização de auditoria contábil e financeira.

OUTPUT:
CLASSIFICAÇÃO: SERVIÇO
CONFIANÇA: Alta
JUSTIFICATIVA: A contratação é para execução de uma atividade específica por
               prestador especializado, caracterizando prestação de serviço.
```

## Stack

| Componente | Tecnologia |
|---|---|
| Framework de fine-tuning | [Unsloth](https://github.com/unslothai/unsloth) |
| Modelo base | Mistral 7B Instruct v0.3 (4-bit) |
| Trainer | TRL `SFTTrainer` |
| Geração do dataset | Mistral API (mistral-small-latest) |
| Ambiente | Google Colab — GPU T4 (15.6GB VRAM) |

## Pipeline (passo a passo)

### 1. Instalação
Instala Unsloth, TRL, Transformers, Accelerate e o cliente Mistral.

### 2. Geração do Dataset
- Define uma lista de itens reais de compras públicas
- Chama a API Mistral para rotular cada item automaticamente
- Salva o dataset no formato **ChatML** (JSONL), com `system`, `user` e `assistant`
- Alternativa manual disponível (seção 2b) para quem já tem dados rotulados

Estrutura de cada exemplo:
```json
{
  "messages": [
    {"role": "system",    "content": "Você é um classificador..."},
    {"role": "user",      "content": "Classifique o item abaixo:\n\n<item>"},
    {"role": "assistant", "content": "CLASSIFICAÇÃO: SERVIÇO\nCONFIANÇA: Alta\nJUSTIFICATIVA: ..."}
  ]
}
```

### 3. Carregamento do Modelo
- Mistral 7B Instruct v0.3 quantizado em **4-bit** via `FastLanguageModel`
- VRAM usada: ~4.17GB (sobram ~10GB no T4 para o treino)

Outras opções comentadas no notebook:
- `unsloth/Llama-3.2-3B-Instruct` — mais leve (~2.5GB)
- `unsloth/Qwen2.5-7B-Instruct` — ótimo para português
- `unsloth/Phi-3.5-mini-instruct` — mais econômico (~3GB)

### 4. Configuração do LoRA
LoRA (Low-Rank Adaptation) treina apenas pequenas matrizes adaptadoras inseridas nas camadas existentes, sem alterar os pesos originais.

| Parâmetro | Valor | Motivo |
|---|---|---|
| `r` | 16 | Equilíbrio capacidade/memória |
| `lora_alpha` | 16 | Mantido igual ao `r` (boa prática) |
| `lora_dropout` | 0 | Otimizado pelo Unsloth |
| `target_modules` | q, k, v, o, gate, up, down | Atenção + FFN |
| Parâmetros treináveis | ~42M / 3.8B | ~1.1% do total |

### 5. Treino
```
Épocas:          3
Batch efetivo:   8 (2 por device × 4 acumulação)
Learning rate:   2e-4 (cosine decay)
Precisão:        fp16
Otimizador:      adamw_8bit
Tempo (15 ex.):  58 segundos
Loss final:      2.1132
```

### 6. Inferência
Usa `FastLanguageModel.for_inference` para geração otimizada. Temperature baixa (0.1) para respostas consistentes e determinísticas.

### 7. Avaliação
Calcula acurácia num subconjunto de teste separado. No experimento com 15 exemplos: **10/10 = 100%**.

> ⚠️ Dataset de demonstração muito pequeno — esse resultado não é estatisticamente representativo. Para produção, use 300-500+ exemplos.

### 8. Export

| Opção | Comando/Método | Caso de Uso |
|---|---|---|
| Adaptadores LoRA | `model.save_pretrained("modelo-lora")` | Retreinar ou carregar com Unsloth |
| GGUF (Ollama) | `model.save_pretrained_gguf(..., "q4_k_m")` | Rodar localmente |
| HuggingFace Hub | `model.push_to_hub(repo, token)` | Compartilhar publicamente |

Para usar o GGUF com Ollama:
```bash
ollama create classificador -f ./modelo-gguf/Modelfile
ollama run classificador
```

## Como Rodar

1. Abra o notebook no Google Colab
2. Vá em `Runtime → Change runtime type → T4 GPU`
3. Configure sua `MISTRAL_API_KEY` nos secrets do Colab (`userdata.get`)
4. Execute as células em ordem

Para usar dados próprios, pule a célula de geração e use a seção **2b** com seus itens já rotulados.

## Melhorando o Modelo

**Acurácia abaixo de 90%?**
- Aumente o dataset para 500+ exemplos
- Aumente `num_train_epochs` para 5
- Verifique o balanceamento das classes (~50/50)

**Modelo não segue o formato de saída?**
- Confirme que `SYSTEM_MODELO` é idêntico no treino e na inferência
- Reduza `temperature` para 0.05 na inferência
- Adicione mais exemplos com o formato exato esperado

**Expandir para mais classes (ex: Obra, Locação)?**
- Adicione as novas classes no `SYSTEM_MODELO` e no dataset
- Mantenha o dataset balanceado entre todas as classes
- Considere aumentar `r=32` no LoRA para maior capacidade

## Referências

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Documentação Unsloth](https://docs.unsloth.ai)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Lei 8.666/93](https://www.planalto.gov.br/ccivil_03/leis/l8666cons.htm) — Licitações e Contratos
- [Lei 14.133/21](https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/L14133.htm) — Nova Lei de Licitações
