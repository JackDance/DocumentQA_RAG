# DocumentQA_RAG

## Environment Setup
### Install Milvus
Please refer to the Docker method of installing Milvus online.

### Download Repository
```shell
git clone https://github.com/JackDance/DocumentQA_RAG
```
### Install Python Packages
```shell
conda create -n your_conda_envir_name python=3.10 -y
conda activate your_conda_envir_name
```
```shell
pip install requirments.txt
```
## Modify Configuration
Open `.env`file and modify the following values
```shell
OPENAI_API_KEY1="your openai api key"
EMB_OPENAI_API_BASE="your openai embedding api base"
CHAT_OPENAI_API_BASE="your chat openai api base"
MILVUS_HOST="milvus host ip"
MILVUS_PORT="milvus port, default is 19530"
```

## Knowledge Building
```shell
cd DocumentQA_RAG
python knowledge_building.py --doc_folder "your local document folder path"
```
## Knowledge Retrieval
```shell
cd DocumentQA_RAG
python knowledge_retrieval.py
```
## Sample Result

https://github.com/JackDance/DocumentQA_RAG/assets/46999456/d01d9fa0-8b6d-473d-a23f-f46dcc8a85c9

