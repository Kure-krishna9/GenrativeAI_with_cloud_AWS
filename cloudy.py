import boto3
import json


prompt_data="""  
act as a Shakespear and write a poem on machine Learning
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    " prompt": prompt_data,
    "maxTokens":512,
    "temprature":0.8,
    "topp":0.9
}
body=json.dump(payload)
model_id="ai21.j2-mid-v1"
response=bedrock.invoke_model(
    body=body,
    modelID=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.load(response.get("body").read())
response_text=response_body.get("complations")[0].get("data").get("text")
print(response_text)