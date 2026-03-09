# Contributing to robotmem

Thanks for your interest in contributing to robotmem!

## Getting Started

```bash
git clone https://github.com/robotmem/robotmem.git
cd robotmem
pip install -e ".[dev]"
```

## Running Tests

```bash
# 如果使用 pip install -e ".[dev]" 安装：
pytest tests/ -v

# 如果从源码直接运行：
PYTHONPATH=src pytest tests/ -v
```

## Code Style

- Python 3.10+
- Type hints on all public functions
- Docstrings on public modules and classes

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests for your changes
4. Run `pytest tests/ -v` and ensure all tests pass
5. Commit with a descriptive message
6. Open a Pull Request

## Reporting Issues

Open an issue at https://github.com/robotmem/robotmem/issues with:

- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
