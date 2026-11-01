# Contributing to COSMOVERSE

Thank you for your interest in contributing to **COSMOVERSE**! Whether you are an astrophysicist, cosmologist, machine learning researcher, open-source engineer, or an autonomous AI agent, your contributions are welcome.

---

## 1. Code of Conduct & Scientific Principles
- **Reproducibility First**: Every pull request that introduces a physical or statistical method must include corresponding unit tests and documentation referencing the underlying literature (e.g. arXiv / DOI).
- **Physical Validity**: Numerical models must be validated against analytic benchmarks or established simulation suites (CAMELS, Quijote, etc.).
- **Traceability**: All experiment configurations must record random seeds and parameter bounds.

---

## 2. Development Workflow

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/clevervi/cosmoverse.git
   cd cosmoverse
   ```

2. **Environment Setup (using `uv` or `pip`)**:
   ```bash
   # With uv (recommended)
   uv venv
   uv pip install -e ".[all]"

   # Or standard pip
   pip install -e ".[all]"
   ```

3. **Running Tests**:
   ```bash
   uv run pytest
   ```

4. **Code Quality and Formatting**:
   ```bash
   uv run ruff check .
   uv run ruff format .
   ```

---

## 3. Pull Request Guidelines
- Branch names should be descriptive: `feature/differentiable-pm`, `fix/transfer-function-silk`, `docs/sbi-math`.
- Include a clear description of the scientific rationale or bug fix.
- Ensure all 25+ pytest unit tests pass before requesting review.

---

## 4. Contact
For questions or collaborations:
- **Lead Developer**: clevervi (Adrian Dario V.)
- **Email**: `adriandariov@gmail.com`
- **GitHub**: [github.com/clevervi](https://github.com/clevervi)
