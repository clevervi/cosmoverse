## Qué cambió

<!-- Resumen corto y concreto de los módulos cosmológicos, observacionales o de inferencia modificados. -->

## Por qué

<!-- Motivación científica o técnica según docs/MASTER_DOCUMENT.md. -->

## Cómo se verificó

<!-- Comandos, salida real de pytest, scripts/ci.ps1 o run_paper1_mvp_protocol.py. -->

```text
# pegar salida real de tests / CI local
```

## Checklist

- [ ] Respeta la separación de 3 niveles (1: Física, 2: Observación, 3: Inferencia/Validación).
- [ ] Tests nuevos o actualizados en `tests/`.
- [ ] `uv run ruff check .` y `uv run pytest -v` en verde.
- [ ] `./scripts/ci.ps1` (Windows) o `./scripts/ci.sh` (Unix) en verde.
- [ ] Sin secretos, tokens, cookies ni datos personales.
- [ ] `docs/MASTER_DOCUMENT.md` o `AGENTS.md` actualizado si cambia una interfaz o decisión metodológica.
- [ ] Issue y milestone vinculados.

## Issue

Closes #
