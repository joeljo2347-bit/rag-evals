# Blind grading

Graded by a separate agent that saw only the help center, the questions, the sources each
answer was given and the answers: no expected answers, no model names, items shuffled.

| Model | Correct | Answerable correct | Uncovered refused | Citations OK |
|---|---|---|---|---|
| gpt-oss:20b | 37/40 | 29/32 | 8/8 | 40/40 |
| qwen3:8b | 39/40 | 31/32 | 8/8 | 40/40 |

### Marked down: gpt-oss:20b

- **a20** Adds 'of receipt', which the help center does not state.
- **a29** Adds unsupported claim that the interval is for maintaining performance and safety.
- **a30** Says 45 days overdue; help center says unpaid after 45 days (invoices are due at 30 days).

### Marked down: qwen3:8b

- **a03** Unhedged claim the order can be cancelled; packing only usually happens within 2 hours.
