FROM python:3.10.6-buster

COPY MedAI_App MedAI_App
COPY requirements_prod.txt requirements_prod.txt

RUN pip install --upgrade pip
RUN pip install -r requirements_prod.txt

CMD uvicorn MedAI_App.api.fast:app --host 0.0.0.0
