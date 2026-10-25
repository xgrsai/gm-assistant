# Phase 4: LoRA-донавчання Кіпера (Qwen2.5-3B, QLoRA)

Скрипт `train_unsloth.py` донавчає LoRA-адаптер на SFT-даних Кіпера (стиль наративу та робота з пасажами KNOWLEDGE BASE).
Працює на Colab / Kaggle (одна T4), не на локальній GTX 1660 Ti. Адаптер потім зливається з повноточною базою й
квантується в GGUF окремим скриптом `merge_and_quantize.py` (Colab, CPU), а локально лише підключається в Ollama
(розділ "Сервінг" нижче).

| Що | Значення |
|---|---|
| База для навчання | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (QLoRA, fp16 на T4) |
| HF-близнюк бази (для злиття й GGUF) | `Qwen/Qwen2.5-3B-Instruct` |
| Сервінг | Ollama без `ADAPTER` (його відхиляє Ollama автора): `merge_and_quantize.py` дає два Q4_K_M GGUF одним llama.cpp: FT = LoRA, злита в базу (`keeper-qwen25-3b:sft-v1`), і база без LoRA (`qwen2.5-3b-selfq4:v1`); обидва з офіційним шаблоном `qwen2.5:3b` (`python -m sft.ollama_models`) |
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
6. **Memory probe (необов'язково, `--memory-probe`):** перед `trainer.train()` один крок навчання на НАЙДОВШОМУ
   train-прикладі; пік памʼяті пишеться в `train_meta.json` -> `gates.memory_probe`. За замовчуванням вимкнено: крок
   поза `trainer.train()` може не мати всіх оптимізацій памʼяті Unsloth і показувати OOM там, де справжнє навчання
   вміщується (так було на першому прогоні на T4). Справжній тест — саме навчання.
7. Навчання з чекпоінтами кожні `--save-steps` у `<out>/checkpoints`.
8. Після навчання: `<out>/adapter/` (LoRA + токенізатор; це ОСТАННІЙ крок, а не найкращий за eval loss),
   `<out>/train_meta.json`; необов'язковий `--llama-cpp-dir` ще робить `<out>/adapter.gguf`, але для сервінгу він більше
   не потрібен (див. "Сервінг: злитий GGUF замість ADAPTER").

`train_meta.json`: id і revision бази (та HF-близнюка для GGUF), sha256 даних і весь manifest, гіперпараметри, seed,
версії torch / transformers / unsloth / trl / peft / bitsandbytes, CUDA, GPU, рядок chat template з sha256 (unsloth
і plain transformers, з прапорцем збігу), звіти gates (включно з типом collator), train / eval loss з кривою eval,
`eval_loss_best` і його крок поруч із кроком збереженого адаптера, sha256 `train_unsloth.py` і `gates.py`, git commit,
а за `--llama-cpp-dir` ще commit llama.cpp і revision HF-бази (для злитих GGUF ці факти пише `gguf_meta.json`).

## `--max-len` і памʼять T4

Дефолт `--max-len 8192`: на сервінгу історія може займати до 6000 токенів, до якої додаються системний промпт і
KNOWLEDGE BASE, тож 4096 відсікало б реальні ходи. Length gate лишається страховкою: якщо хоч один приклад довший,
прогін зупиняється до початку навчання. Ціна: довші послідовності = більше активацій і памʼяті на 16 ГБ T4
(batch 1 + gradient checkpointing Unsloth, але логіти по словнику Qwen ~152k зростають разом з довжиною). Чи вміщається
це на T4 з цими налаштуваннями, не перевірено.

Памʼять залежить від РЕАЛЬНОЇ довжини найдовшого прикладу при batch 1, а не від `--max-len` (це лише стеля для
length gate). Тому, якщо НАВЧАННЯ дає OOM, `--max-len` зменшувати марно. Рецепт: перебудувати дані (крок S4) з
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

## Сервінг: злитий (merged) GGUF замість ADAPTER

Ollama автора відхиляє `ADAPTER` ("LoRA adapters are no longer supported"), тому `adapter.gguf` (конвертація
`convert_lora_to_gguf.py`, llama.cpp `71255fef…`) **не використовується** для жодного результату: він лишається лише в
`gguf.superseded` model card. Натомість (запасний варіант з `docs/critique.md` C4) `merge_and_quantize.py` зливає LoRA в
повноточну `Qwen/Qwen2.5-3B-Instruct` (float16) і квантує в Q4_K_M, а **базу** конвертує й квантує тим самим llama.cpp
без LoRA. Тож арми base і FT відрізняються лише навчанням (база НЕ є офіційним blob `qwen2.5:3b`, а власним Q4_K_M).

| Ollama-тег | Що це | GGUF |
|---|---|---|
| `qwen2.5-3b-selfq4:v1` | база: Qwen2.5-3B-Instruct, сконвертована й проквантована без LoRA | `qwen2.5-3b-instruct.Q4_K_M.gguf` |
| `keeper-qwen25-3b:sft-v1` | FT: LoRA злита в базу, потім Q4_K_M | `keeper-qwen25-3b-sft-v1.Q4_K_M.gguf` |
| `qwen2.5:3b` | офіційна модель Ollama: лише джерело шаблону й stop-токенів, не арм | (вже завантажена) |

LoRA тренувалась проти NF4-бази, а зливається в float16-базу: звичайна практика QLoRA, невелика розбіжність
train/serve, яку треба згадати в тексті роботи (`merge.note` у `gguf_meta.json`).

### Крок 1. Злиття і квантування (Colab, CPU-рантайм)

Runtime -> Change runtime type -> **CPU** (GPU тут не потрібен). Після `pip install -r requirements-merge.txt`
виконайте `!pip uninstall -y torchao`: Colab має torchao 0.10, і PEFT через неї відмовляється завантажувати адаптер
(злиттю torchao не потрібна; скрипт перевіряє це на старті). Команди, залежності (`requirements-merge.txt`) і
порядок перевірок описані в докстрінгу `training/merge_and_quantize.py` (`--adapter-dir` = тека `adapter/` з результату
навчання, `--out` = тека на Drive). Скрипт зупиняється, якщо злиття не змінило ваг (адаптер-no-op, C4), якщо хеші не
збігаються тощо; наприкінці він записує `gguf_meta.json` (commit llama.cpp, версії peft/transformers, dtype злиття,
sha256 обох GGUF, sha256 адаптера) і друкує наступні кроки.

### Крок 2. Файли на машину з Ollama

Покладіть у git-ignored `adapters/keeper-qwen25-3b-sft-v1/` (поруч із результатами навчання `adapter/`,
`train_meta.json`, `manifest.json`):

```
adapters/keeper-qwen25-3b-sft-v1/
    adapter/adapter_model.safetensors  adapter/adapter_config.json   # з навчання
    train_meta.json  manifest.json                                   # з навчання
    keeper-qwen25-3b-sft-v1.Q4_K_M.gguf                              # з merge_and_quantize.py
    qwen2.5-3b-instruct.Q4_K_M.gguf                                  # з merge_and_quantize.py
    gguf_meta.json                                                   # з merge_and_quantize.py
```

### Крок 3. Modelfile і `ollama create`

База і FT мають мати **той самий** офіційний шаблон чату й stop-токени. `sft.ollama_models` бере Modelfile офіційної
`qwen2.5:3b` (як `ollama show qwen2.5:3b --modelfile`), міняє в ньому лише рядок `FROM` на `FROM ./<файл>.gguf` і пише
`Modelfile.base` та `Modelfile.ft` поруч із GGUF. Нічого не додавайте вручну (`ADAPTER`, `TEMPLATE`, `SYSTEM`):

```
cd src/backend
uv run python -m sft.ollama_models modelfiles --gguf-dir ../../adapters/keeper-qwen25-3b-sft-v1
```

`qwen2.5:3b` має бути в `ollama list`; робіть `ollama pull qwen2.5:3b` **лише якщо її там немає**: повторний pull може
оновити офіційну модель, на якій працюють арми phase3, посеред дослідження. Команда друкує `ollama create` для обох
моделей; виконуйте їх з теки з GGUF і повертайтесь у `src/backend` (команди без `&&`, по одній у рядок, бо в PowerShell 5.1
`&&` немає):

```
cd ../../adapters/keeper-qwen25-3b-sft-v1
ollama create qwen2.5-3b-selfq4:v1 -f Modelfile.base
ollama create keeper-qwen25-3b:sft-v1 -f Modelfile.ft
cd ../../src/backend
```

Імена GGUF беруться з `gguf_meta.json` (`outputs.<роль>.file`), якщо квантування було не Q4_K_M; прапорці `--base-gguf`,
`--ft-gguf`, `--base-tag`, `--ft-tag`, `--official` перевизначають усе.

## Перевірка адаптера і model card

Критика C4: модель може створитись без помилок і не відрізнятися від бази (адаптер "мовчки" не діє), а `ollama run` не має
`temperature` / `seed` і не передає системний промпт Кіпера, тож відмінності там нічого не доводять. Тому
`sft.check_adapter` ходить напряму в `/api/chat` Ollama з `temperature: 0`, `seed: 0`, `num_ctx: 16384`, на 20
детермінованих прикладах `artifacts/sft/coc7e-v1/val.jsonl` (повідомлення без останньої репліки асистента; НЕ gold-дані
оцінювання). Після кроку 3 (обидва `ollama create` виконано з теки `adapters/keeper-qwen25-3b-sft-v1`) поверніться в
`src/backend`: `check_adapter` запускається звідти (шляхи за замовчуванням відраховуються від кореня репозиторію, а модуль
`sft` лежить у `src/backend`):

```
cd ../../src/backend
uv run python -m sft.check_adapter
```

Прогін триває ≈15 хв на GTX 1660 Ti (base x20, base ще раз x5, FT x20), прогрес друкується. Кожен запит іде з
`keep_alive: 0`, тобто з холодного старту: Ollama повторно використовує закешований префікс промпту (спільний системний
промпт), а промпт, обчислений від префікса іншої довжини, на CUDA дає трохи інші числа, і жадібне декодування
розходиться через кілька десятків токенів (перший прогін 2026-10-25: 3 з 5 повторів base розійшлись; два холодні запити
того самого промпту дали однакові відповіді). Перезавантаження моделі додає кілька секунд на запит. Гейти (будь-який
провалений = код виходу 1, повідомлення пояснює, що робити, а результати й картка все одно записуються):

| Гейт | Що перевіряє |
|---|---|
| `determinism` | перші 5 промптів base, надіслані вдруге (після повного першого проходу, кожен запит з холодного старту), дають ту саму відповідь; інакше порівняння з FT безглузде |
| `effect` | FT-відповідь відрізняється від base-відповіді для ≥ 80% промптів (порожня відповідь не рахується) |
| `template` | шаблон чату base = FT = офіційна `qwen2.5:3b` (`/api/show`) |
| `weights` | blob за тегом = sha256 відповідного GGUF з теки; blob-и base і FT різні; жодна модель не має шару `ADAPTER` |
| `parity` | **строго base проти FT** (це пара, яку порівнює робота): заголовки обох GGUF мають однакову кількість тензорів, а `model_info` (плюс family / розмір / квантування) з `/api/show` збігається, окрім ключів `general.*` (інші можна виключити `--parity-ignore ПРЕФІКС` після огляду). Різниця в параметрах конвертації (rope base, довжина контексту, id EOS) змішала б ефект навчання з артефактом конвертації |

Порівняння `model_info` base з офіційною `qwen2.5:3b` лише **інформативне**: її blob давно сконвертував інший llama.cpp,
тож безневинні розбіжності метаданих очікувані. Вони пишуться в `serving.parity.official_differences` картки й
друкуються попередженням, але не валять гейт і не знижують статус (шаблон чату проти офіційного, навпаки, перевіряється
строго: його копіює Modelfile). Порожній шаблон чату вважається значенням (він відрізняється від справжнього), а не
"невідомо". Не перевірене через API (старий Ollama без `model_info` у base/FT, немає `qwen2.5:3b` для порівняння шаблону)
позначається `NOT VERIFIED` і не валить прогін, але **статус картки тоді `passed_unverified`, а не `passed`**; `evals/model_cards.py` приймає лише `passed` (інакше потрібен
`--allow-failed-card`), тож виправте те, що не вдалось прочитати (наприклад, `ollama pull qwen2.5:3b`, якщо її немає в
`ollama list`), і повторіть перевірку. Явна розбіжність дає `failed`. Інформативно (не гейти): частка відповідей із рядком 🎲 там, де є кидок (і хибних там, де його немає),
частка кирилиці в україномовних відповідях, частка відповідей з `[R#]`, довжина відповіді; дельти FT-base, FT-еталон
(таргет у val.jsonl), base-еталон. Перед запуском перевіряється, що `val.jsonl` — це саме файл, яким навчали (sha256 з
`manifest.json` і з `train_meta.json`), і що `section_split.v1.json` не змінювали.

Результат:

- `adapters/keeper-qwen25-3b-sft-v1/check_adapter.json` — повний результат із кожною відповіддю (git-ignored). Якщо
  Ollama впав посеред прогону, отримані відповіді потрапляють у **окремий** `check_adapter.incomplete.json` з
  `"incomplete": true` (код виходу 2, картки немає): наявний повний `check_adapter.json`, на який посилається
  `results_file.sha256` закоміченої картки, не перезаписується; наступний повний прогін видаляє залишок. `--card-only`
  відхиляє неповний файл і файл зі старою схемою результатів (поточна `schema: 3`: у 2 додано `status`, у 3 холодні
  запити, `settings.keep_alive`);
- `training/model_cards/keeper-qwen25-3b-sft-v1.json` — **model card** (комітиться; її читає `evals/model_cards.py`, а
  прогони оцінювання відмовляються стартувати, якщо digest тегу в Ollama інший). У ній: гіперпараметри, версії,
  втрати, зведення гейтів і sha256 chat template з `train_meta.json`; sha256 `adapter_model.safetensors`; sha256 обох GGUF,
  commit llama.cpp, dtype злиття й весь `gguf_meta.json` дослівно; digest-и Ollama (`ollama_digests.ft/base`, `serving.*`)
  і результат порівняння шаблонів; sha256 `train.jsonl` / `val.jsonl` і `manifest.json`, `section_split.v1.json`; git
  commit даних і поточний commit; результати перевірки без відповідей.

`status` картки: `passed` (усі гейти пройдені й перевірені), `passed_unverified` (пройдені, але щось не вдалось
перевірити або була **пропущена** перевірка узгодженості злиття: `waived_mismatches` у `gguf_meta.json` від
`merge_and_quantize.py --allow-mismatch`, чи адаптер не збігається з тим, що записано в `train_meta.json`; усе це лежить
у верхньому полі картки `waived_mismatches` і друкується попередженням), `failed`. `--card-only` перебудовує лише картку з уже збережених результатів (без Ollama), `--n`,
`--repeat`, `--num-predict`, `--min-differ` змінюють параметри прогону. Якщо `effect` провалено — це і є сигнал C4: злиття не змінило поведінки або тег
створено не з того файлу; якщо `determinism` — закрийте інші програми, що використовують GPU/модель, і повторіть.

## Застереження

- Адаптер тренується на NF4-базі (QLoRA), а зливається з float16-базою і далі квантується в Q4_K_M. Це невелика
  розбіжність train/serve, яку треба згадати в тексті роботи. Base-арм (`qwen2.5-3b-selfq4:v1`) проходить той самий
  конвеєр convert + quantize без LoRA, тож він не збігається з офіційним blob `qwen2.5:3b`, але відрізняється від FT лише
  навчанням.
- Не комітьте ваги, адаптери та `.gguf` (див. `.gitignore`); у git лише код, README і model card
  (`training/model_cards/*.json`). Modelfile генерується (`python -m sft.ollama_models`), а не зберігається в репозиторії.
- Не змінюйте промпт, дані чи модель посеред серії: кожна зміна даних = нова версія (`--version v2`) і новий `--out`.
