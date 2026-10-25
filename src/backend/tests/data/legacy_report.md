# Eval report: syn

Fingerprint `ffffffffffffffff` · datasets {'rules_qa': 'fa286aeace419e57', 'conflict_probes': '83d158c33c8510af', 'adjudication': 'e04abe4242a1a694'} · judge `{'rules_qa': 'judge|judge-qa-v1|aaaa', 'conflict_probes': 'judge|judge-conflict-v1|bbbb'}`

## Retrieval (section-level gold, top-k, no quotas/threshold)

| query | n | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | unanswerable top score |
|---|---|---|---|---|---|---|---|
| en | 10 | 0.900 | 0.900 | 0.900 | 0.800 | 0.900 | – |

| query | question type | n | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|---|
| en | lookup | 10 | 0.900 | 0.900 | 0.900 |

### min_score calibration (over the top 4 hits the production retriever returns)

| query |  | threshold | precision | recall | F1 | F2 | unanswerable with no hits |
|---|---|---|---|---|---|---|---|
| en | highest keeping 95% recall | 0.500 | 0.450 | 0.800 | 0.576 | 0.692 | – |
| en | best F2 | 0.500 | 0.450 | 0.800 | 0.576 | 0.692 | – |
| en | best F1 | 0.500 | 0.450 | 0.800 | 0.576 | 0.692 | – |

## Rules QA

| arm | n | score | 95% CI | strict acc | key facts | contradicts | unanswerable ok | confident wrong | lang ok | gold in context | cites gold | invalid [R#] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 12 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | – | – | – |
| rag | 12 | 0.900 | [0.750, 1.000] | 0.800 | 0.900 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.250 |

### Paired comparisons (answerable items)

| comparison | n | Δ score | 95% CI | p (sign-flip) | p (Holm) | only base correct | only arm correct | McNemar p |
|---|---|---|---|---|---|---|---|---|
| rag vs prompt | 10 | 0.900 | [0.750, 1.000] | 0.002 | 0.002 | 0 | 8 | 0.008 |

## Knowledge-conflict probes

| arm | context adherence | parametric override | control accuracy | parametric knowledge (no context) |
|---|---|---|---|---|
| prompt | – | – | – | 1.000 |
| rag | 0.500 | 0.500 | 1.000 | – |

| arm | perturbation/familiarity | n | context adherence |
|---|---|---|---|
| rag | negation/well_known | 1 | 1.000 |
| rag | numeric/obscure | 1 | 0.000 |

## Adjudication

| arm | clean full | needs_roll | roll type | target | false roll | missed roll | injected full | flip rate | harm rate |
|---|---|---|---|---|---|---|---|---|---|
| prompt | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| rag | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

| arm | injection type | harm rate |
|---|---|---|
| prompt | claimed_roll | 0.000 |
| rag | claimed_roll | 0.000 |

| comparison | only base correct | only arm correct | McNemar p |
|---|---|---|---|
| rag vs prompt | 0 | 0 | 1.000 |
