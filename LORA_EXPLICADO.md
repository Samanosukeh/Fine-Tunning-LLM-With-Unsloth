# LoRA Fine-tuning — Parâmetros e Arquitetura Explicados

## Pipeline completo do projeto

```mermaid
flowchart TD
    A[Itens brutos de licitação] --> B[Mistral API\nmistral-small-latest]
    B --> C[dataset.jsonl\nformato ChatML]
    C --> D[SFTTrainer]
    E[Mistral 7B\n3.8B params\nCONGELADO] --> D
    F[Adaptadores LoRA\n42M params\nTREINÁVEIS] --> D
    D --> G{Export}
    G --> H[LoRA weights\n~100MB\nretreino]
    G --> I[GGUF q4_k_m\nOllama local]
    G --> J[HuggingFace Hub\npúblico]
```

---

## Arquitetura de um Transformer (base de tudo)

Todo LLM moderno é uma pilha de **camadas repetidas**. Cada camada tem dois blocos principais:

```mermaid
flowchart TD
    IN[Token de entrada] --> ATT

    subgraph ATT["🔍 Bloco de Atenção"]
        A1[q_proj · k_proj · v_proj] --> A2[Scores Q × K]
        A2 --> A3[Softmax → pondera V]
        A3 --> A4[o_proj]
    end

    ATT --> N1[Add & Norm]
    IN  --> N1

    N1 --> FFN

    subgraph FFN["⚙️ Bloco FFN — SwiGLU"]
        F1[gate_proj → SiLU]
        F2[up_proj]
        F3["× multiplicação"]
        F4[down_proj]
        F1 --> F3
        F2 --> F3
        F3 --> F4
    end

    FFN --> N2[Add & Norm]
    N1  --> N2
    N2  --> OUT[Próxima camada]
```

O Mistral 7B tem **32 dessas camadas** empilhadas.

---

## Bloco de Atenção — o que é e por que importa

A atenção resolve: *"dado o token atual, quais outros tokens do contexto são relevantes?"*

```mermaid
flowchart LR
    X[Token\nde entrada]

    X -->|copia| Q[q_proj\nQuery\n'o que procuro?']
    X -->|copia| K[k_proj\nKey\n'o que ofereço?']
    X -->|copia| V[v_proj\nValue\n'informação a extrair']

    Q --> S["Q × Kᵀ\nscores de atenção"]
    K --> S
    S --> SM[Softmax\nnormaliza pesos]
    SM --> W["soma ponderada\ncom V"]
    V --> W
    W --> O[o_proj\nOutput\nprojeta resultado]
    O --> Y[saída da atenção]
```

**Por que incluir no LoRA?**
É aqui que o modelo aprende *relações semânticas*. Para o classificador de licitações, é a atenção que aprende que "contratação de empresa para..." tem padrão diferente de "aquisição de 500 unidades de...".

---

## Bloco FFN — o que é e por que importa

Depois da atenção, a FFN processa cada token **individualmente** — é onde ficam armazenados os "fatos" e padrões do modelo.

No Mistral usa arquitetura **SwiGLU** com 3 matrizes:

```mermaid
flowchart LR
    X[x]

    X --> G["gate_proj\n4096 → 14336"]
    X --> U["up_proj\n4096 → 14336"]

    G --> S["SiLU\nativa ou bloqueia"]
    S --> M["× multiplicação\nelemento a elemento"]
    U --> M

    M --> D["down_proj\n14336 → 4096"]
    D --> Y[saída]
```

- `gate_proj` — decide *quais* features "abrir passagem"
- `up_proj` — expande a representação para espaço maior (4096 → 14336)
- `down_proj` — comprime de volta para a dimensão original (14336 → 4096)

**Por que incluir no LoRA?**
A FFN armazena conhecimento factual. Incluir essas camadas ensina ao modelo os padrões específicos do domínio — vocabulário jurídico de licitações, critérios da Lei 8.666/93, etc.

---

## Como o LoRA funciona na prática

Sem LoRA, para adaptar o modelo você teria que atualizar **cada uma** dessas matrizes enormes (ex: `q_proj` tem dimensão 4096×1024 = ~4M parâmetros).

O LoRA **não toca** os pesos originais. Em vez disso, insere dois adaptadores minúsculos em paralelo:

```mermaid
flowchart LR
    X[entrada\n4096 dims]

    X --> W["W_original\n4096 × 1024\n🔒 CONGELADO"]
    X --> A["A\n4096 × r\n✏️ TREINÁVEL"]

    A --> B["B\nr × 1024\n✏️ TREINÁVEL"]

    W --> SUM["➕"]
    B --> SUM

    SUM --> Y["saída\n1024 dims\n= W·x + A·B·x"]
```

> Com `r=16`, as matrizes A e B juntas têm `4096×16 + 16×1024 = 82.432` parâmetros — contra os `4.194.304` da matriz original. **Redução de 98%.**

O resultado final é `W_original(x) + A×B(x)`. Só `A` e `B` são atualizados.

---

## Os parâmetros do notebook explicados

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    bias="none",
    random_state=42,
)
```

### `r` — o mais importante

O rank controla o tamanho das matrizes `A` e `B`. Quanto maior, mais expressivo e mais memória.

| Valor | Parâmetros treináveis | Quando usar |
|---|---|---|
| `r=8` | ~20M | Dataset pequeno, tarefa simples |
| `r=16` | ~42M | **Equilíbrio geral (usado aqui)** |
| `r=32` | ~84M | Tarefa complexa, múltiplas classes |
| `r=64` | ~167M | Mudança de comportamento profunda |

### `lora_alpha` — a escala

Controla o quanto o adaptador influencia o modelo original. A fórmula aplicada é:

```
saída = W_original(x) + (alpha / r) × A×B(x)
```

Com `alpha=r=16`, o fator fica `16/16 = 1.0` — influência neutra, sem amplificar nem suprimir. É a configuração mais segura e mais comum.

### `lora_dropout`

Dropout aplicado nos adaptadores durante o treino para regularização. O Unsloth recomenda `0` porque já aplica outras otimizações internas que tornam o dropout redundante.

### `use_gradient_checkpointing="unsloth"`

Em vez de manter todos os gradientes na VRAM durante o backprop, recomputa os intermediários sob demanda. Economiza ~40% de VRAM no T4, permitindo usar `max_seq_length=2048` sem OOM.

### `bias="none"`

Não treina os vetores de bias — raramente necessário e economiza parâmetros.

### `target_modules` — onde inserir os adaptadores

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # atenção (4 × 32 camadas)
    "gate_proj", "up_proj", "down_proj",       # FFN    (3 × 32 camadas)
]
```

Isso resulta em `7 módulos × 32 camadas = 224 pares de adaptadores (A, B)`.

**Por que não incluir tudo?** Camadas como embeddings e layer norms são mais sensíveis e raramente precisam ser ajustadas para fine-tuning de domínio.

---

## Parâmetros do SFTTrainer

```python
SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    fp16=True,
    optim="adamw_8bit",
)
```

| Parâmetro | Valor | Motivo |
|---|---|---|
| `num_train_epochs` | 3 | Suficiente para datasets pequenos sem overfitting |
| `per_device_train_batch_size` | 2 | Limite da VRAM disponível no T4 |
| `gradient_accumulation_steps` | 4 | Batch efetivo = 2×4 = 8, sem gastar mais VRAM |
| `learning_rate` | 2e-4 | Padrão recomendado para LoRA |
| `warmup_steps` | 10 | Sobe o LR gradualmente nas primeiras steps |
| `lr_scheduler_type` | cosine | Decaimento suave até o fim do treino |
| `fp16` | True | T4 não tem suporte a bf16, usa fp16 |
| `optim` | adamw_8bit | Adam quantizado — mesma qualidade, menos VRAM |

**Batch efetivo:** o modelo nunca vê os 8 exemplos de uma vez. Ele processa 2 de cada vez e acumula os gradientes por 4 steps antes de atualizar os pesos. O efeito matemático é equivalente a um batch de 8.

---

## Resumo visual — o que é treinado

```mermaid
flowchart TD
    subgraph MODELO["Mistral 7B — 3.800.305.664 parâmetros"]
        EMB["Embedding layer\n🔒 CONGELADO"]

        subgraph C1["Camada 1 de 32 (mesma estrutura em todas)"]
            subgraph ATT1["Atenção"]
                Q1["q_proj 🔒 + LoRA ✏️"]
                K1["k_proj 🔒 + LoRA ✏️"]
                V1["v_proj 🔒 + LoRA ✏️"]
                O1["o_proj 🔒 + LoRA ✏️"]
            end
            subgraph FFN1["FFN SwiGLU"]
                G1["gate_proj 🔒 + LoRA ✏️"]
                U1["up_proj   🔒 + LoRA ✏️"]
                D1["down_proj 🔒 + LoRA ✏️"]
            end
        end

        LMH["LM Head\n🔒 CONGELADO"]
        DOTS["· · · 32 camadas no total · · ·"]
    end

    STATS["✏️ Treináveis: 41.943.040 params\n🔒 Congelados: 3.758.362.624 params\n📊 Proporção: 1.10% do total"]

    MODELO --> STATS
```

---

## Referências

- [LoRA: Low-Rank Adaptation of Large Language Models (paper original)](https://arxiv.org/abs/2106.09685)
- [Unsloth — documentação](https://docs.unsloth.ai)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Mistral 7B — arquitetura SwiGLU/GQA](https://arxiv.org/abs/2310.06825)
