FROM python:3.8

# Copy and install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app /app

# Use working directory /api
ENV PYTHONPATH=/app
WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]