// Ukrainian UI strings and label maps.

export const ERROR_MESSAGES = {
  not_found: 'Кампанію не знайдено — можливо, її вже видалено.',
  turn_in_progress: 'Хранитель ще відповідає на попередню дію. Зачекайте кілька секунд і спробуйте знову.',
  duplicate_turn: 'Цю дію вже було надіслано. Історію оновлено.',
  turn_incomplete: 'Попередній хід перервався. Повторіть його або відкиньте, щоб продовжити гру.',
  nothing_to_retry: 'Немає перерваного ходу, який можна повторити. Історію оновлено.',
  empty_generation: 'Хранитель не зміг сформулювати відповідь (порожня генерація). Спробуйте ще раз.',
  timeout: 'Модель не відповіла вчасно.',
  rate_limited: 'Вичерпано ліміт запитів до моделі. Зачекайте хвилину й повторіть.',
  upstream_unavailable: 'Сервіс моделі тимчасово недоступний.',
  internal_error: 'Внутрішня помилка сервера. Подробиці — у журналі сервера.',
  connection_lost: 'З’єднання обірвалося посеред ходу.',
  network_error: 'Не вдалося зв’язатися з сервером. Перевірте, чи запущено бекенд.',
  server_unavailable: 'Сервер не відповідає. Перевірте, чи запущено бекенд.',
  validation_error: 'Сервер відхилив запит: некоректні дані.',
  unknown_arm: 'Сервер не знає обраного режиму Хранителя. Список режимів оновлено — оберіть режим ще раз.',
}

export function errorMessage(code) {
  if (!code) return 'Сталася невідома помилка.'
  return ERROR_MESSAGES[code] ?? `Сталася помилка (${code}).`
}

// `progress.node` → status line under the streaming Keeper bubble.
export const STAGE_LABELS = {
  prepare_turn: 'Хранитель вислуховує дію…',
  analyze: 'Хранитель формулює запит…',
  retrieve: 'Хранитель гортає правила…',
  adjudicate: 'Хранитель зважує дію…',
  mechanics: 'Кидок кубиків…',
  narrate: 'Хранитель розповідає…',
}
export const STAGE_DEFAULT = 'Хранитель слухає…'

export function stageLabel(node) {
  return (node && STAGE_LABELS[node]) || STAGE_DEFAULT
}

export const DIFFICULTY_LABELS = {
  regular: 'Звичайна',
  hard: 'Складна',
  extreme: 'Екстремальна',
}

export const LEVEL_LABELS = {
  critical: 'Критичний успіх',
  extreme: 'Екстремальний успіх',
  hard: 'Складний успіх',
  regular: 'Успіх',
  failure: 'Провал',
  fumble: 'Фатальний провал',
}

export const ROLL_KIND_LABELS = {
  skill: 'Навичка',
  characteristic: 'Характеристика',
  luck: 'Удача',
  sanity: 'Розсудливість',
}

export const CHARACTERISTIC_LABELS = {
  STR: 'Сила',
  CON: 'Витривалість',
  SIZ: 'Статура',
  DEX: 'Спритність',
  APP: 'Зовнішність',
  INT: 'Інтелект',
  POW: 'Воля',
  EDU: 'Освіта',
}

// Retrieved knowledge-base chunks cited by the Keeper as `[R<n>]`.
export const SOURCE_TYPE_LABELS = {
  rules: 'Правила',
  lore: 'Лор світу',
}

export const CITATION_INVALID_HINT = 'Цього фрагмента не було серед знайдених — можлива галюцинація'

/** Human label of an experiment arm (`meta.arms[]`). */
export function armLabel(arm, arms = []) {
  if (!arm) return ''
  const base = arm.use_rag ? 'З базою правил (RAG)' : 'Без бази знань'
  const twins = arms.filter((a) => Boolean(a.use_rag) === Boolean(arm.use_rag))
  return twins.length > 1 && arm.generator_model ? `${base} · ${arm.generator_model}` : base
}

export const LANGUAGE_LABELS = {
  uk: 'Українська',
  en: 'English',
}

/** Scenario fields are `{uk, en}` objects; tolerate plain strings too. */
export function localized(value, lang = 'uk') {
  if (value == null) return ''
  if (typeof value === 'string') return value
  return value[lang] ?? value.uk ?? value.en ?? Object.values(value)[0] ?? ''
}
