# build image

docker build --tag beigene-async .

# run container

docker run -dt -v $PWD/app/backend/static:/beigene-chunk/app/backend/static:ro --publish 5001:5001 --name beigene-async
beigene-async

# 接口文档

```shell
http://127.0.0.1:5000/docs/api/
```

##### sso attributes
python3-saml==1.15.0
```json
{
  'http://schemas.microsoft.com/claims/authnmethodsreferences': [
    'urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport'
  ],
  'http://schemas.microsoft.com/identity/claims/displayname': [
    'Hongbo Liu'
  ],
  'http://schemas.microsoft.com/identity/claims/identityprovider': [
    'https://sts.windows.net/7dbc552d-50d7-4396-aeb9-04d0d393261b/'
  ],
  'http://schemas.microsoft.com/identity/claims/objectidentifier': [
    'c2e8ed79-c802-4d90-b56a-78a977b74151'
  ],
  'http://schemas.microsoft.com/identity/claims/tenantid': [
    '7dbc552d-50d7-4396-aeb9-04d0d393261b'
  ],
  'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress': [
    'hongbo1.liu@beigene.com'
  ],
  'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname': [
    'Hongbo'
  ],
  'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name': [
    'hongbo1.liu@beigene.com'
  ],
  'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname': [
    'Liu'
  ]
}

```

# 定时任务 Qinglong

```shell
docker run -dit -v $PWD/ql/data:/ql/data -p 5700:5700 -e QlBaseUrl="/" --name qinglong --hostname qinglong --restart unless-stopped whyour/qinglong:debian
```

#### qinglong 账号密码

```text
beigene beigene123
```

#### 链接：

> http://bg-chatbot-qa.westus2.cloudapp.azure.com:5700/


# 法国
dfc23a65cf9e40cea71911cf93ca444e
https://apac-openai-service-fc.openai.azure.com/
gpt35turbo
gpt4






https://adalconsole-dev.beigene.cn/aem/assetpreview?assetPath=/content/dam/beigene/《百济神州公司发言人及新闻活动制度》_2022年8月_Final.pdf
https://adalconsole-dev.beigene.cn/aem/assetpreview?assetPath=/content/dam/beigene/高危神经母细胞瘤治疗的标准方案+2.pdf
https://adalconsole-dev.beigene.cn/aem/assetpreview?assetPath=/content/dam/beigene/scientific/PD3ew2.pdf
https://adalconsole-dev.beigene.cn/aem/assetpreview?assetPath=/content/dam/beigene/pdf-Chinses.pdf