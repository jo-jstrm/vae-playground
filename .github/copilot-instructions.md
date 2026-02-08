# Guidelines for Variational Autoencoder Training

## Style

- Use dependency injection for all components, including the model, optimizer, and data loaders. 
This allows for greater flexibility and easier testing.
- Use type hints for all functions and methods to improve code readability and maintainability.
- Follow PEP 8 style guidelines for Python code, including naming conventions and formatting.
- Use docstrings to document all functions and classes, including their parameters and return values.
Use ReStructuredText format for docstrings to ensure consistency and readability.
- No need for logging libs, prefer print statements for simplicity.

## Tools 

- Use `uv run` to execute any python commands, including ruff and pytest. 
- Use ruff
- Ther is an mlflow docker container running on localhost 8080.
