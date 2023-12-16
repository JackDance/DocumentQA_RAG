#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：beigene-async
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/6/29 15:48 
# ====================================
import logging
import mimetypes
import os
import json
import time
from logging.handlers import TimedRotatingFileHandler
from urllib.parse import quote
from typing import Union

import openai
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.responses import RedirectResponse
from fastapi import HTTPException, Path, Request
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from flask import redirect
from pymilvus import MilvusException
from sqlalchemy.orm import scoped_session
from starlette.templating import Jinja2Templates
from werkzeug.utils import redirect

from app.approaches.question import QuestionDecomposeApproach
from app.approaches.language import LanguageDetectApproach, AzureLanguageDetectApproach
from app.approaches.konwledge import KnowledgeApproach
from app.azureai import AzureOpenAI
from app.database.session import Session
from app.exceptions import OpenAITimeoutException, TranslatorException
from app.schemas.chat import Chat
from app import utils
from app.approaches.faq import FAQApproach, FAQPlanApproach
from app.database.models import RatingsModel
from onelogin.saml2.auth import OneLogin_Saml2_Auth, OneLogin_Saml2_Utils
# 加载环境变量
from app.schemas.rating import Rating, RatingCancel

from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import JSONResponse

load_dotenv(".azure/rg-apac-demo/.env")

# Replace these with your own values, either in environment variables or directly here
AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT") or "mystorageaccount"
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER") or "content"
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE") or "gptkb"
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "gptkbindex"
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or "myopenai"
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "davinci"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "chat"
KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY") or "category"
KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE") or "sourcepage"

# azure openai的调用方式
os.environ["OPENAI_API_KEY"] = "2e3bc8a0624246b9b5a679f11b5ce5cb"
openai.api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
openai.api_type = "azure"
# openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_base = "https://apac-openai-service-eus.openai.azure.com/"
openai.api_version = "2023-03-15-preview"

app = FastAPI()

# Azure Translator
# Add your key and endpoint
AZURE_TRANSLATOR_KEY = "b6426b30ee3d4270b6bc6ef769382ac5"
AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"

# 密钥用于加密会话数据
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Azure AD SAML设置
saml_settings = {
    "strict": True,
    "debug": True,
    "sp": {
        "entityId": "spn:69946be3-3193-40c4-94e3-e82f818f10d9",
        "assertionConsumerService": {
            "url": "https://bg-chatbot-qa.westus2.cloudapp.azure.com/sso/",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
        },
        "singleLogoutService": {
            "url": "https://bg-chatbot-qa.westus2.cloudapp.azure.com/logout",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
        },
        "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified",
        "x509cert": "Your SP x509 Certificate",
        "privateKey": "Your SP Private Key"
    },
    "idp": {
        "entityId": "https://sts.windows.net/7dbc552d-50d7-4396-aeb9-04d0d393261b/",
        "singleSignOnService": {
            "url": "https://login.microsoftonline.com/7dbc552d-50d7-4396-aeb9-04d0d393261b/saml2",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
        },
        "singleLogoutService": {
            "url": "https://login.microsoftonline.com/7dbc552d-50d7-4396-aeb9-04d0d393261b/saml2",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
        },
        "x509cert": "MIIC8DCCAdigAwIBAgIQNaELrQkFIItD8SLgUWc1vjANBgkqhkiG9w0BAQsFADA0MTIwMAYDVQQDEylNaWNyb3NvZnQgQXp1cmUgRmVkZXJhdGVkIFNTTyBDZXJ0aWZpY2F0ZTAeFw0yMzA2MDkwMzI1NTJaFw0yNjA2MDkwMzI1NTFaMDQxMjAwBgNVBAMTKU1pY3Jvc29mdCBBenVyZSBGZWRlcmF0ZWQgU1NPIENlcnRpZmljYXRlMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1r4YfAm0P6+kVO70Wrbxx8LYRv4B7QvMhKtdZGdxJsy29ikmF6GBC5otPnNPV/4CEYRrLO8vNgMbrf5wozXrKVp1PHdBqIAXrdGVx2eDlzhPCp6/PBvabYj6YTRYdfPGDIkMYZBsXm+w6jB59876QYLRboixkQXEpZcVTR7PQdbTivmBtr/GFgELF0Kk/YsV+54k5ZtHBsXMkKZ8yMA3heFUBkzlNlb+aNMWLoQJuvb6bW8AHECeC6qIMjDO91b+FUbIkKlo/e0h9uYDyGlTVwd8M+WMVYfqYGLmQHQFPdlLmH3WmCNsgocyPkgakG61Dj1cf6FoPUTYr9RlwY7hWQIDAQABMA0GCSqGSIb3DQEBCwUAA4IBAQANmHcLN5LdKXZyZW2+TrKz8Wzl00ibM0ej6YWeD2vuP3yyS07rxxoIJ+UrMacLtByZjAXCIbhcc1qN8Y8zYzz6JspS2E2XfTQDnnqgYcmSSEAriIVV5XxhAQoTgje2cJAaJFjy3a8SFDqq6iSSnvFdxs0FjyMjbLmNftqEhvhhJ5lya8cqb9d9UgKrgtv+6O9TteEbjDHl7S8qAt8Vt3K/gieutFLPS0MNHxZJ3O1wJ8a6j9xc05FgSzI+My10K/4cGBGbWRM9YRDhbQ2xOyZjdZAPY3vkPUEmhqSiZPSw/vqSsB+c9YZiuhfGzoM3Sp5xkSm4Lsw3qNMX9sNWJT8T"
    }
}

account_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
shared_access_key = os.getenv("AZURE_STORAGE_ACCESS_KEY")
credential = shared_access_key

# Create the BlobServiceClient object
blob_client = BlobServiceClient(account_url, credential=credential)
blob_container = blob_client.get_container_client(AZURE_STORAGE_CONTAINER)

app.mount("/static", StaticFiles(directory="static"), name="static")

OPENAI_API_KEY = "YOUR_API_KEY"
OPENAI_API_URL = "https://api.openai.com/v1/engines/davinci-codex/completions"

faq_args = {"connection_args": {"host": "20.115.226.41", "port": "19530"},
            "collection_name": "faq",
            "text_field": "question"}

knowledge_args = {"connection_args": {"host": "20.115.226.41", "port": "19530"},
                  # "collection_name": "document_chunk_ip",
                  "collection_name": "knowledge",
                  "text_field": "chunk"}

embeddings = OpenAIEmbeddings()
faq_datastore = Milvus(embedding_function=embeddings, collection_name=faq_args["collection_name"],
                       connection_args=faq_args["connection_args"], text_field=faq_args["text_field"])
knowledge_datastore = Milvus(embedding_function=embeddings, collection_name=knowledge_args["collection_name"],
                             connection_args=knowledge_args["connection_args"], text_field=knowledge_args["text_field"])

# 法国没有 embedding 模型
embedding_endpoints = [
    # (endpoint, key, model_name)
    ("https://apac-openai-service-eus.openai.azure.com", "2e3bc8a0624246b9b5a679f11b5ce5cb",
     "text-embedding-ada-002"),
    ("https://apac-openai-service-scus.openai.azure.com", "afc31eb73f474752b5ea7e1a4613b4a8",
     "text-embedding-ada-002"),
]
# 法国3.5模型部署名称和美区不一致
gpt35_chat_endpoints = [
    # (endpoint, key, model_name)
    ("https://apac-openai-service-eus.openai.azure.com", "2e3bc8a0624246b9b5a679f11b5ce5cb", "chat"),
    ("https://apac-openai-service-scus.openai.azure.com", "afc31eb73f474752b5ea7e1a4613b4a8", "chat"),
    ("https://apac-openai-service-fc.openai.azure.com", "dfc23a65cf9e40cea71911cf93ca444e", "gpt35turbo")
]
#
gpt4_chat_endpoints = [
    # (endpoint, key, model_name)
    ("https://apac-openai-service-eus.openai.azure.com", "2e3bc8a0624246b9b5a679f11b5ce5cb", "gpt4"),
    ("https://apac-openai-service-fc.openai.azure.com", "dfc23a65cf9e40cea71911cf93ca444e", "gpt4")
]
# Azure ai Endpoint
azure_ai_endpoint = AzureOpenAI(embedding_endpoints, gpt35_chat_endpoints, gpt4_chat_endpoints)
# FAQ
faq = FAQPlanApproach(faq_datastore)
# Knowledge
knowledge = KnowledgeApproach(knowledge_datastore, azure_ai_endpoint)
# 语言类型检测
# language_detect = LanguageDetectApproach()

language_detect = AzureLanguageDetectApproach(AZURE_TRANSLATOR_ENDPOINT, AZURE_TRANSLATOR_KEY)
# 用户问题拆解
question_decompose = QuestionDecomposeApproach()


@app.exception_handler(OpenAITimeoutException)
async def openai_exception_handler(request: Request, exc: OpenAITimeoutException):
    return JSONResponse(
        status_code=200,
        content={"code": 504, "msg": "openai服务器响应超时！", "data": None}
    )


@app.exception_handler(TranslatorException)
async def translator_exception_handler(request: Request, exc: TranslatorException):
    return JSONResponse(
        status_code=200,
        content={"code": 504, "msg": "azure translator 服务器响应超时！", "data": None}
    )


@app.exception_handler(MilvusException)
async def milvus_exception_handler(request: Request, exc: MilvusException):
    return JSONResponse(
        status_code=200,
        content={"code": 504, "msg": "Milvus服务器响应超时！", "data": None}
    )


# SSO
@app.get('/sso/login')
async def sso_login(request: Request):
    # SAML登录请求路由
    req = await utils.prepare_flask_request(request)
    saml_auth = OneLogin_Saml2_Auth(req, saml_settings)
    return RedirectResponse(saml_auth.login())


@app.post('/sso/')
async def sso_acs(request: Request):
    # SAML ACS (Assertion Consumer Service)路由
    req = await utils.prepare_flask_request(request)
    saml_auth = OneLogin_Saml2_Auth(req, saml_settings)

    session = request.session

    request_id = None
    if 'AuthNRequestID' in session:
        request_id = session['AuthNRequestID']

    saml_auth.process_response(request_id=request_id)
    errors = saml_auth.get_errors()
    # not_auth_warn = not saml_auth.is_authenticated()
    if len(errors) == 0 and saml_auth.is_authenticated():
        if 'AuthNRequestID' in session:
            del session['AuthNRequestID']
        session['samlUserdata'] = saml_auth.get_attributes()
        session['samlNameId'] = saml_auth.get_nameid()
        session['samlNameIdFormat'] = saml_auth.get_nameid_format()
        session['samlNameIdNameQualifier'] = saml_auth.get_nameid_nq()
        session['samlNameIdSPNameQualifier'] = saml_auth.get_nameid_spnq()
        session['samlSessionIndex'] = saml_auth.get_session_index()
        # # return redirect("/")
        # return RedirectResponse("/")
        html_content = f"""
        <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/">
        </head>
        <body>
            <p>Redirecting to a new URL...</p>
        </body>
        </html>
        """

        # 返回 HTML 响应并进行重定向
        return HTMLResponse(content=html_content, status_code=200)

        # self_url = OneLogin_Saml2_Utils.get_self_url(req)
        # if 'RelayState' in request.form and self_url != request.form['RelayState']:
        # To avoid 'Open Redirect' attacks, before execute the redirection confirm
        # the value of the request.form['RelayState'] is a trusted URL.
        # return redirect(saml_auth.redirect_to(request.form['RelayState']))
    elif saml_auth.get_settings().is_debug_active():
        error_reason = saml_auth.get_last_error_reason()
        error = {"error": error_reason}
        return {"code": 200, "msg": "ok", "data": error}


@app.get('/sso/logout')
async def sso_logout(request: Request):
    # SAML登出请求路由
    session = request.session

    req = await utils.prepare_flask_request(request)
    saml_auth = OneLogin_Saml2_Auth(req, saml_settings)
    saml_auth.logout()
    session.clear()
    return RedirectResponse('/sso/index')  # 重定向到首页


@app.get('/sso/profile')
def sso_profile(request: Request):
    session = request.session

    if 'samlUserdata' in session:
        attributes = session['samlUserdata']
        # 在这里处理仪表板页面的逻辑和展示用户的SAML属性
        displayname = attributes.get('http://schemas.microsoft.com/identity/claims/displayname')
        emailaddress = attributes.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress')

        profile = {
            "authenticated": True,
            "displayName": displayname[-1] if len(displayname) >= 1 else None,
            "emailAddress": emailaddress[-1] if len(emailaddress) >= 1 else None
        }
        res = {"code": 200, "msg": "ok", "data": profile}
        return res
    else:
        # return redirect('/sso/login')
        profile = {
            "authenticated": False,
            "displayName": None,
            "emailAddress": None
        }
        res = {"code": 200, "msg": "ok", "data": profile}
        return res


@app.get("/content/{path:path}")
def content_file(path: str = Path(..., regex=".+")):
    blob = blob_container.get_blob_client(path).download_blob()
    mime_type = blob.properties.content_settings.content_type

    if mime_type == "application/octet-stream":
        mime_type, _ = mimetypes.guess_type(path)
        mime_type = mime_type or "application/octet-stream"

    content = blob.readall()

    path = quote(path)

    return Response(content, media_type=mime_type, headers={
        "Content-Disposition": f'inline; filename="{path}"'
    })


@app.post("/chat")
async def async_chat(chat: Chat):
    approach = chat.approach
    history = chat.history
    overrides = chat.overrides.dict() or {}
    model = chat.model  # gpt3.5 (chat) gpt4 (gpt4)
    print(f"history: {history}")
    print(f'{"*" * 50} START {"*" * 50}')

    # 检测语言
    language = await language_detect.run(history, overrides, model)
    overrides["language"] = language

    # 单个原始问题拆解为多个子问题
    sub_question_list = await question_decompose.run(history, overrides, model)

    faq_status, result = faq.run(history, overrides, model, sub_question_list=sub_question_list)
    if faq_status:
        print(f"该问题匹配到FAQ库")
        pass
    else:
        # result = knowledge.run(history, overrides, sub_question_list=sub_question_list)
        result = await knowledge.run(history, overrides, model, sub_question_list=[history[-1]["user"]])
        print("该问题匹配到Knowledge库")

    # 前端语言显示
    result["verbose"] = utils.language_verbose(language)

    print(f'{"*" * 50} END {"*" * 50}')

    res = {"code": 200, "msg": "ok", "data": result}

    return res

@app.post("/chat_streaming")
async def async_chat_streaming(chat: Chat):
    approach = chat.approach
    history = chat.history
    overrides = chat.overrides.dict() or {}
    model = chat.model  # gpt3.5 (chat) gpt4 (gpt4)

    print(f'{"*" * 50} START {"*" * 50}')
    print(f"history: {history}")
    # 检测语言
    language = await language_detect.run(history, overrides, model)
    overrides["language"] = language

    # 单个原始问题拆解为多个子问题
    sub_question_list = await question_decompose.run(history, overrides, model)

    faq_status, faq_result = faq.run(history, overrides, model, sub_question_list=sub_question_list)
    if faq_status:
        print(f"该问题匹配到FAQ库")
        def faq_event_generator():
            all_content = ""
            result = {
                "data_points": faq_result['data_points'],
                "thoughts": faq_result['thoughts'],
                "verbose": utils.language_verbose(language)
            }
            for ans in faq_result["answer"]:
                print(ans, end="", flush=True)
                all_content += ans
                result["answer"] = all_content
                data_string = json.dumps({"code": 200, "msg": "ok", "data": result})
                res_dict_str = f'data: {data_string}\r\n\r\n'
                yield res_dict_str
                time.sleep(0.01)
        faq_event = faq_event_generator()
        return StreamingResponse(faq_event, media_type="text/event-stream")
    else:
        print("该问题匹配到Knowledge库")
        # >>>>> start <<<<<<
        # 修改过的run函数的返回结果，类型是dict
        running_res = await knowledge.run_streaming(history, overrides, model, sub_question_list=[history[-1]["user"]])  # history[-1]["user"]为用户当前问的问题，不加入之前问过的问题和答案

        # azure openai environment
        # openai.api_type = "azure"
        # openai.api_key = "2e3bc8a0624246b9b5a679f11b5ce5cb"
        # openai.api_base = "https://apac-openai-service-eus.openai.azure.com/"
        # openai.api_version = "2023-03-15-preview"

        def get_chat_completion_stream(
                messages: list[dict],
                engine="gpt-35-turbo",
                temperature: Union[int, float] = 0,
        ):
            res = openai.ChatCompletion.create(
                messages=messages,
                temperature=temperature,
                engine=engine,
                stream=True
            )

            all_content = ""
            result = {
                "datapoints": running_res['datapoints'],
                "thoughts": running_res['thoughts'],
                "source2adal_urls": running_res['source2adal_urls'],
                "verbose": utils.language_verbose(language)
            }

            for event in res:
                if "content" in event["choices"][0].delta:
                    current_response = event["choices"][0].delta.content
                    print(current_response, end="", flush=True)
                    # build final streaming output
                    all_content += current_response
                    result["answer"] = all_content
                    data_string = json.dumps({"code": 200, "msg": "ok", "data": result})
                    res_dict_str = f'data: {data_string}\r\n\r\n'

                    yield res_dict_str

        chat_prompts = [
            {"role": "user", "content": running_res["prompt"]}
        ]
        try:
            get_chat_completion_stream(chat_prompts)
            print(f'{"*" * 50} END {"*" * 50}')
            return StreamingResponse(
                get_chat_completion_stream(chat_prompts),
                media_type="text/event-stream",
            )
        except openai.error.Timeout:
            print(f"OpenAI API request timed out")
            data_string = json.dumps({"code": 408, "msg": "Server Connection Timeout", "data": None})
            res_dict_str = f'data: {data_string}\r\n\r\n'
            print(f'{"*" * 50} END {"*" * 50}')
            return JSONResponse(content=res_dict_str)
        except openai.error.APIConnectionError:
            print(f"OpenAI APIConnection Error")
            data_string = json.dumps({"code": 443, "msg": "OpenAI APIConnection Error", "data": None})
            res_dict_str = f'data: {data_string}\r\n\r\n'
            print(f'{"*" * 50} END {"*" * 50}')
            return JSONResponse(content=res_dict_str)
        except:
            # 捕获内部服务器错误
            print("Internal Server Error")
            data_string = json.dumps({"code": 500, "msg": "Internal Server Error", "data": None})
            res_dict_str = f'data: {data_string}\r\n\r\n'
            print(f'{"*" * 50} END {"*" * 50}')
            return JSONResponse(content=res_dict_str)


@app.get("/")
async def index(request: Request):
    # session = request.session
    # if 'samlUserdata' in session:
    #     attributes = session['samlUserdata']
    #     return FileResponse("static/index.html")
    # else:
    #     return RedirectResponse('/sso/login')
    # 本地调试
    return FileResponse("static/index.html")

@app.get("/demo")
async def demo(request: Request):
    return FileResponse("static/index.html")


@app.post("/rating")
def rating_route(rating: Rating):
    # 请求数据
    request_json = rating.dict()

    # 数据验证
    # 创建会话对象
    db_session = Session()

    new_record = RatingsModel(**request_json)

    db_session.add(new_record)
    db_session.commit()
    db_session.refresh(new_record)

    res = {"code": 200, "msg": "ok", "data": {"id": new_record.id}}
    return res


@app.delete("/rating_cancel")
def rating_cancel_route(rating: RatingCancel):
    # 创建会话对象
    db_session = Session()
    # 请求数据
    id = rating.id
    record = db_session.query(RatingsModel).filter(RatingsModel.id == id).one()
    db_session.delete(record)
    db_session.commit()

    res = {"code": 200, "msg": "ok", "data": None}

    return res


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
