FROM python:3.10-slim

COPY api.py .
COPY project.py .
COPY requirements.txt .
COPY artifacts/ ./artifacts/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
#CMD python project.py && evidently ui --host 0.0.0.0 --port 8082
