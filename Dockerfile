FROM python:3.9

WORKDIR /NER

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]