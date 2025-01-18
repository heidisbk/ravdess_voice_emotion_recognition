FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY project.py .

# Lancer project.py puis lancer "evidently ui"
# On suppose que tu veux le dashboard interactive sur http://localhost:8082
CMD python project.py && evidently ui --host 0.0.0.0 --port 8082
