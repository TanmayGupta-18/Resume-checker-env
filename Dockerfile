FROM python:3.11-slim

WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi uvicorn pydantic requests openai python-multipart openenv-core

# Expose port
EXPOSE 7860

# Run the server using uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
