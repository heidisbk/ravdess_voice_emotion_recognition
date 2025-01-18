FROM python:3.10-slim

WORKDIR /app

# Installer curl pour les requêtes HTTP
RUN apt-get update && apt-get install -y curl && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY project.py .

# Commande unique pour démarrer l'interface UI, attendre sa disponibilité, puis exécuter le script
CMD bash -c "\
    evidently ui --host 0.0.0.0 --port 8082 --workspace evidently_workspace & \
    until curl -s http://localhost:8082/api/version; do \
        echo 'Waiting for Evidently UI to be ready...'; \
        sleep 2; \
    done; \
    python project.py"
