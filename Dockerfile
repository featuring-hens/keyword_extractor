FROM python:3.12

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root

COPY . .

CMD ["poetry", "run", "uvicorn", "image_to_keywords:app", "--host", "0.0.0.0", "--port", "8000"]
