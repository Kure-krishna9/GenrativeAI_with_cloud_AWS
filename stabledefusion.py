import boto3
import json
import base64
import os

prompr_data="""   
provided me an 4k hd image of a beach,also use a blue sky rainyseason and cinematic display"""
prompt_template=[{"text":prompr_data,"weight":1}]
bedrock=boto3.client(service_name="bedrock_runtime")
payload={
    "text_prompts":prompt_template,
    "cfg_scale":10,
    "seed":0,
    "steps":50,
    "width":512,
    "height":512
}

body=json.dump(payload)
model_id="stability.stable-diffusion-x1-v0" \
""
response=bedrock.invoke_model(
    body=body,
    modelID=model_id,
    accept="application/json",
    contentType="application/json"
)


response_body=json.load(response.get("body").read())
artifect=response_body.get("artifacts")[0]
image_encoded=artifect.get("base64").encode("utf-8")
image_bytes=base64.b64decode(image_encoded)

# save image
output_dir="output"

os.makedirs(output_dir,exist_ok=True)
file_name=f"{output_dir}genrated_img.jpg"
with open(file_name,"wb")as f:
    f.write(image_bytes)
# response_text=response_body.get("complations")[0].get("data").get("text")
# print(response_text)