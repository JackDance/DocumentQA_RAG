FROM python:3.10.1

RUN  apt-get update
RUN  apt install -y pkg-config libxml2-dev libxmlsec1-dev libxmlsec1-openssl build-essential

WORKDIR /beigene-async

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

EXPOSE 5001

COPY . .

CMD ["uvicorn","main:app" , "--workers", "4", "--host", "0.0.0.0", "--port", "5001"]