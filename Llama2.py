import boto3
import json


prompt_data="""  
act as a Shakespear and write a poem on machine Learning
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    " prompt":"[INST]"+ prompt_data +"[/INST]",
    "max_gen_len":512,
    "temprature":0.5,
    "top_p":0.9
}

body=json.dump(payload)
model_id="meta.llama2-70b-chat-v1"
response=bedrock.invoke_model(
    body=body,
    modelID=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.load(response.get("body").read())
response_text=response_body['generation']
print(response_text)
