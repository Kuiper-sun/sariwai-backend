FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 user
USER user

# Tell transformers to use a cache folder the user can write to
ENV TRANSFORMERS_CACHE="/home/user/.cache"

ENV PATH="/home/user/.local/bin:$PATH"

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]