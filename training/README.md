# Phase 4: LoRA-донавчання Кіпера (Qwen2.5-3B, QLoRA)

Скрипт `train_unsloth.py` донавчає LoRA-адаптер на SFT-даних Кіпера (стиль наративу та робота з пасажами KNOWLEDGE BASE).
Працює на Colab / Kaggle (одна T4), не на локальній GTX 1660 Ti. Локально адаптер лише конвертується в GGUF і
підключається в Ollama.

| Що | Значення |
|---|---|
| База для навчання | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (QLoRA, fp16 на T4) |
| HF-близнюк бази (для GGUF) | `Qwen/Qwen2.5-3B-Instruct` |
| Сервінг | Ollama: `FROM qwen2.5:3b` + `ADAPTER adapter.gguf` (той самий Q4_K_M blob, що й у base-арм) |
| Дані | `artifacts/sft/coc7e-v1/{train,val}.jsonl` + `manifest.json` (кроки `python -m sft.build`, S4) |
| Піни | `unsloth==2026.10.2`, `unsloth_zoo==2026.10.2`, `trl==1.13.0`, `peft==0.20.0` (`requirements-train.txt`) |

Чому саме `unsloth==2026.10.2`: `unsloth==2026.9.4` вимагає `trl<=0.24.0` і не ставиться разом із `trl==1.13.0`.
torch / transformers / bitsandbytes беруться з образу Colab/Kaggle; після першого успішного прогону їхні версії
треба взяти з `train_meta.json` (`versions`) і закріпити в `requirements-train.txt`.

Скрипт писався проти цих пінів і не запускався на GPU під час написання: перший прогін перевіряйте за розділом
"Що перевірити".

## Що робить скрипт (і де зупиняється)

1. sha256 `train.jsonl` / `val.jsonl` має збігатися з `manifest.json` (`files.<name>.sha256`), інакше відмова.
2. Кожен приклад рендериться власним chat template токенізатора. Системний промпт береться з даних (шаблон Qwen
   підставляє "You are Qwen..." лише коли system-повідомлення немає; gate це перевіряє). Хвіст `<|im_end|>\n`
   обрізається до `<|im_end|>`, щоб TRL не дописував другий EOS.
3. **Length gate:** довжина рахується точним токенізатором. Якщо хоч один приклад довший за `--max-len`, прогін
   зупиняється (інакше TRL мовчки відрізав би відповідь).
4. **Маскування: loss лише на ОСТАННІЙ репліці Кіпера.** У системному промпті лежать KNOWLEDGE BASE, кидки й
   адʼюдикація лише ПОТОЧНОГО ходу, тож репліки історії не є валідними таргетами під цим промптом (модель вчилась би
   називати результати кидків і `[R#]`, яких немає в контексті, а ранні ходи переважали б у loss). Тому скрипт
   викликає `train_on_responses_only` (теги за сімейством: Qwen `<|im_start|>user\n` / `<|im_start|>assistant\n`;
   Gemma 4 `<|turn>user\n` / `<|turn>model\n`), а потім перераховує labels: навчається все після ОСТАННЬОГО
   response-тегу, усе раніше = -100.
5. **Masking gate** (на тому, що побачить тренер): у кожному прикладі рівно один неперервний supervised-відрізок, він
   закінчується на останньому токені, і `decode(відрізок) == останнє повідомлення асистента + EOS`; токенізований
   приклад закінчується рівно одним EOS; labels, які реально віддає data collator, збігаються з перевіреними (якщо
   collator тренера їх ігнорує або падає, скрипт ставить `DataCollatorForSeq2Seq` і записує це в meta). Дані мають бути
   NFC-нормалізовані, а відповіді асистента без пробілів/переводів рядка на краях (токенізатор Qwen нормалізує до NFC;
   `check_messages` відхиляє інакше, повідомлення gate показує індекс першої розбіжності з контекстом).
6. **Memory probe:** перед `trainer.train()` один forward+backward на НАЙДОВШОМУ train-прикладі; пік памʼяті
   (`torch.cuda.max_memory_allocated`) пишеться в `train_meta.json` -> `gates.memory_probe`; при OOM скрипт зупиняється
   з поясненням.
7. Навчання з чекпоінтами кожні `--save-steps` у `<out>/checkpoints`.
8. Після навчання: `<out>/adapter/` (LoRA + токенізатор; це ОСТАННІЙ крок, а не найкращий за eval loss),
   `<out>/train_meta.json`, за `--llama-cpp-dir` ще `<out>/adapter.gguf`.

`train_meta.json`: id і revision бази (та HF-близнюка для GGUF), sha256 даних і весь manifest, гіперпараметри, seed,
версії torch / transformers / unsloth / trl / peft / bitsandbytes, CUDA, GPU, рядок chat template з sha256 (unsloth
і plain transformers, з прапорцем збігу), звіти gates (включно з типом collator), train / eval loss з кривою eval,
`eval_loss_best` і його крок поруч із кроком збереженого адаптера, sha256 `train_unsloth.py` і `gates.py`, git commit,
а для GGUF ще commit llama.cpp і revision HF-бази.

## `--max-len` і памʼять T4

Дефолт `--max-len 8192`: на сервінгу історія може займати до 6000 токенів, до якої додаються системний промпт і
KNOWLEDGE BASE, тож 4096 відсікало б реальні ходи. Length gate лишається страховкою: якщо хоч один приклад довший,
прогін зупиняється до початку навчання. Ціна: довші послідовності = більше активацій і памʼяті на 16 ГБ T4
(batch 1 + gradient checkpointing Unsloth, але логіти по словнику Qwen ~152k зростають разом з довжиною). Чи вміщається
це на T4 з цими налаштуваннями, не перевірено.

Памʼять залежить від РЕАЛЬНОЇ довжини найдовшого прикладу при batch 1, а не від `--max-len` (це лише стеля для
length gate). Тому, якщо memory probe дає OOM, `--max-len` зменшувати марно. Рецепт: перебудувати дані (крок S4) з
меншим бюджетом історії як НОВУ версію даних (`--version v2`, новий `--out`), і лише потім зменшувати `--r`.
`--check-data` покаже розподіл довжин без GPU.

## Підготовка даних (локально)

```
cd src/backend
uv run python -m sft.build build --version v1
```

Результат: `artifacts/sft/coc7e-v1/{train.jsonl,val.jsonl,manifest.json}`. Завантажте ці три файли на Google Drive
в `MyDrive/gm-assistant/sft/coc7e-v1/` (Kaggle: як Dataset).

Дешева перевірка даних перед платною/лімітованою GPU-сесією (потрібні лише `transformers` і доступ до HF, без GPU).
Вона друкує і sha256 chat template; повний прогін порівнює його з шаблоном unsloth-токенізатора:

```
python training/train_unsloth.py --check-data --train artifacts/sft/coc7e-v1/train.jsonl --val artifacts/sft/coc7e-v1/val.jsonl --out artifacts/sft/_unused
```

## Colab

1. Runtime -> Change runtime type -> **T4 GPU**.
2. Комірка 1: Drive і код.

```
from google.colab import drive
drive.mount("/content/drive")
```

```
# варіант А: клон репозиторію
!git clone --depth 1 https://github.com/xgrsai/gm-assistant.git /content/gm-assistant
# варіант Б: без репозиторію; завантажте на Drive training/ та src/backend/sft/gates.py
# !mkdir -p /content/gm-assistant && cp -r /content/drive/MyDrive/gm-assistant/training /content/gm-assistant/
# !cp /content/drive/MyDrive/gm-assistant/gates.py /content/gm-assistant/training/gates.py
```

3. Комірка 2: залежності (після цього, за потреби, Runtime -> Restart session).

```
!pip install -r /content/gm-assistant/training/requirements-train.txt
```

4. Комірка 3: навчання. Вихід одразу на Drive, щоб пережити розрив сесії.

```
!python /content/gm-assistant/training/train_unsloth.py \
    --train /content/drive/MyDrive/gm-assistant/sft/coc7e-v1/train.jsonl \
    --val /content/drive/MyDrive/gm-assistant/sft/coc7e-v1/val.jsonl \
    --out /content/drive/MyDrive/gm-assistant/out/keeper-qwen25-3b-sft-v1
```

5. Якщо сесія обірвалась: знову монтуємо Drive, ставимо залежності і запускаємо **ту саму команду з `--resume`**.
   Скрипт відмовиться, якщо конфігурація (дані, seed, гіперпараметри) не збігається з тією, з якої почався прогін,
   і якщо не знайде `run_config.json` ні в `<out>`, ні в останньому чекпоінті. Без `--resume` при наявних
   чекпоінтах скрипт теж зупиняється, щоб не змішати два прогони.
6. Результати лежать у `MyDrive/gm-assistant/out/keeper-qwen25-3b-sft-v1/`: `adapter/`, `train_meta.json`,
   `run_config.json`, `manifest.json`, `checkpoints/`.

## Kaggle (альтернатива)

1. New Notebook -> Settings: **Accelerator GPU T4** (P100 теж пропонується, але bitsandbytes 4-bit на Pascal не
   гарантований; T4 x2 не потрібен, скрипт використовує одну карту), **Internet: On** (pip і Hugging Face).
2. Add Data -> Upload: створіть Dataset із `train.jsonl`, `val.jsonl`, `manifest.json`; він з'явиться як
   `/kaggle/input/<dataset>/`.
3. Код: клон репозиторію або Dataset із папкою `training/` та `gates.py`.

```
!git clone --depth 1 https://github.com/xgrsai/gm-assistant.git /kaggle/working/gm-assistant
!pip install -r /kaggle/working/gm-assistant/training/requirements-train.txt
```

```
!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/gm-assistant/training/train_unsloth.py \
    --train /kaggle/input/<dataset>/train.jsonl --val /kaggle/input/<dataset>/val.jsonl \
    --out /kaggle/working/out/keeper-qwen25-3b-sft-v1
```

4. Результат у `/kaggle/working/out/...` зберігається лише після **Save Version -> Save & Run All (Commit)**; завантажте
   його з вкладки Output. Сесія обмежена за часом, тож для довгого прогону після розриву покладіть `checkpoints/`
   в Dataset, розпакуйте в `--out/checkpoints` і запустіть з `--resume`. Скопіюйте також `run_config.json` у `--out`
   (він є і в кожній теці `checkpoint-N`, тож без нього скрипт візьме копію з останнього чекпоінта; без жодної копії
   `--resume` відмовляє).

## Чек-лист першого прогону на Colab

Лише справжній прогін на GPU може підтвердити (скрипт на GPU не запускався):

- `pip install -r training/requirements-train.txt` резолвиться без конфліктів (unsloth 2026.10.2 / trl 1.13.0 / peft 0.20.0).
- `tokenizer.eos_token` = `<|im_end|>` (id 151645), а не `<|endoftext|>`: інакше `check_rendered` / `check_token_ends` впадуть.
- Chunked cross-entropy Unsloth справді працює, пік памʼяті з `gates.memory_probe` у `train_meta.json` реалістичний для T4.
- `padding_free` не ввімкнувся мовчки (TRL/unsloth можуть його вмикати): collator gate має показати очікуваний тип і
  збіг labels; `fallback: true` у `gates.collator` означає, що колатор тренера не зберіг наші labels.
- Наші labels пережили видалення колонок TRL: у логу `masking gate ok [train/val] (final reply only)` і правильний
  блок "supervised tokens of example 0".
- `run_config.json` зʼявляється в `checkpoints/checkpoint-N` (потрібно для `--resume` з одним лише `checkpoints/`).
- `chat_template.same_as_transformers` не `false`.

## Орієнтовний час

Для ~1000 прикладів (900 train), 2 епохи, T4: орієнтовно 1-3 години (оцінка, не вимір; з `--max-len 8192` і довгими
прикладами ближче до верхньої межі). Точніше видно за `it/s` у перших кроках. Ліміти free-сесій (Colab ~ кілька годин,
Kaggle ~ 30 год GPU на тиждень) вкладаються, але для Colab розраховуйте на один-два `--resume`.

## Що перевірити

- У лозі є `manifest ok`, `length gate ok [train]`, `length gate ok [val]`, `masking gate ok [train] (final reply only)`,
  `masking gate ok [val] (final reply only)`, `collator gate ok`. Якщо gate впав, GPU-час не витрачено: виправте дані чи
  `--max-len`.
- Блок "supervised tokens of example 0" містить **лише останню відповідь Кіпера** + `<|im_end|>`, без системного
  промпта, без джерел і без історії.
- Блок "first rendered example" починається з НАШОГО системного промпта (не "You are Qwen...").
- `train_meta.json` -> `losses`: `eval_loss_decreased: true`, `eval_loss_last` нижчий за `eval_loss_first`; train loss
  без розривів. Якщо eval loss росте з першої оцінки, зменшіть `--lr` (наприклад 5e-5) або `--epochs 1`. Адаптер
  зберігається з останнього кроку: якщо `eval_loss_best_step` суттєво раніший за `adapter.saved_at_step`, це ознака
  перенавчання (порівняйте `eval_loss_best` і `final_eval_loss`).
- `train_meta.json` -> `versions`, `hardware`, `chat_template` (поле `same_as_transformers` не `false`), `code`
  заповнені (потрібні для розділу про відтворюваність).

## Адаптер у GGUF

`convert_lora_to_gguf.py` з llama.cpp потребує конфіг ПОВНОЇ бази (`--base`), а не 4-бітного репозиторію з
`adapter_config.json`. Це легко робити локально на Windows (ваги бази не потрібні, лише json-файли); краще в окремому venv,
бо requirements llama.cpp можуть змінити версії torch / transformers.

```
git clone --depth 1 https://github.com/ggml-org/llama.cpp
python -m venv .venv-llamacpp
.venv-llamacpp/Scripts/pip install -r llama.cpp/requirements/requirements-convert_lora_to_gguf.txt huggingface_hub
.venv-llamacpp/Scripts/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-3B-Instruct', allow_patterns=['*.json', 'merges.txt']))"
git -C llama.cpp rev-parse HEAD
```

Команда `snapshot_download` друкує шлях до теки з конфігом бази (`<BASE_CFG>`); `git rev-parse` дає commit llama.cpp, який варто
записати поруч з адаптером. Далі, з `<OUT>` = тека результатів з Drive:

```
.venv-llamacpp/Scripts/python llama.cpp/convert_lora_to_gguf.py --base <BASE_CFG> --outtype f16 --outfile <OUT>/adapter.gguf <OUT>/adapter
```

Те саме може зробити скрипт наприкінці навчання, якщо передати `--llama-cpp-dir <шлях до llama.cpp>` (sha256 GGUF,
commit llama.cpp і revision HF-бази потрапляють у `train_meta.json`). Лінукс/Colab: ті самі команди з
`.venv-llamacpp/bin/`.

## Підключення в Ollama

Результат кладемо в git-ignored `adapters/` у корені репозиторію (`.gguf` та `adapters/` ігноруються git):

```
mkdir adapters/keeper-qwen25-3b-sft-v1
cp <OUT>/adapter.gguf <OUT>/train_meta.json <OUT>/manifest.json adapters/keeper-qwen25-3b-sft-v1/
cp training/ollama/Modelfile.template adapters/keeper-qwen25-3b-sft-v1/Modelfile
cd adapters/keeper-qwen25-3b-sft-v1
ollama pull qwen2.5:3b
ollama create keeper-qwen25-3b:sft-v1 -f Modelfile
```

## Швидка перевірка, що адаптер діє

Критика C4: адаптер може завантажитись без помилок і не змінювати виходи. `ollama run` не має прапорців
`temperature` / `seed` і не передає системний промпт Кіпера, тож відмінності між моделями там нічого не доводять.
Порівнюйте через `/api/chat` з детермінованим декодуванням на реальних прикладах `val.jsonl` (повідомлення без
останньої репліки асистента: `[system, ...історія, user]`). Ефект зараховується, лише якщо FT-відповідь відрізняється
від base-відповіді ПРИ `temperature: 0, seed: 0`, а сама base при двох запусках дає однаковий результат (інакше
порівняння безглузде). `num_ctx` має вміщати найдовший промпт, інакше Ollama мовчки обріже його.

```
import json
import urllib.request

def chat(model, messages):
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": 16384, "num_predict": 300}}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["message"]["content"]

with open("artifacts/sft/coc7e-v1/val.jsonl", encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh][:5]
differ = stable = 0
for row in rows:
    prompt = row["messages"][:-1]
    base, base_again = chat("qwen2.5:3b", prompt), chat("qwen2.5:3b", prompt)
    ft = chat("keeper-qwen25-3b:sft-v1", prompt)
    stable += base == base_again
    differ += base != ft
print(f"base deterministic {stable}/{len(rows)}, FT differs from base {differ}/{len(rows)}")
```

Очікування: `differ` близько `len(rows)`. `0` означає, що адаптер не застосовано: перевірте `ollama show --modelfile
keeper-qwen25-3b:sft-v1` (є рядок `ADAPTER`), шлях до `adapter.gguf` і що GGUF зроблено з `--base` повної бази.
Повна перевірка (більше прикладів, метрика стилю) з'явиться як `python -m sft.check_adapter` (крок S8).

## Застереження

- Адаптер тренується на NF4-базі (QLoRA), а Ollama застосовує його до Q4_K_M. Це невелика розбіжність
  train/serve, яку треба згадати в тексті роботи; base і FT при сервінгу мають однаковий blob.
- Не комітьте ваги, адаптери та `.gguf` (див. `.gitignore`); у git лише код, README і `Modelfile.template`.
- Не змінюйте промпт, дані чи модель посеред серії: кожна зміна даних = нова версія (`--version v2`) і новий `--out`.
