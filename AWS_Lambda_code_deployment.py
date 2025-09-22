import boto3
import botocore.config
import json
import response
from datetime import datetime
def blog_genration_using_bedrock(blogtopic:str)->str:
    prompt=f""" 
     <s>[INST]Human:write a 200 words blog on the topic {blogtopic}
    Assistant:[/INST]

"""
    body={
    "prompt":prompt,
    "max_gen_len":512,
    "temprature":0.8,
    "top_p":0.9

    }

    try:
        bedrock=boto3.client("bedrock-runtime",regain_name="us-east-1",config=botocore.config(read_timeout=300,retries={'max_attempts':3}))

        bedrock.invoke_model(body=json.dumps(body),modelId="meta.llama2-13b-chat-v1")
        response_content=response.get('body').read()
        response_data=json.load(response_content)
        print(response_data)
        blog_details=response_data['generation']
        return blog_details
        
    except Exception as e:
        print(f"Error genrating the blog{e}")
        return " "

def save_blog_details_s3(s3_key,s3_bucket,genrate_blog):


    s3=boto3.client('s3')

    try:
        s3.put_object(Bucket=s3_bucket,key=s3_key,Body=genrate_blog)
        print("code save to s3")

    except Exception as e:
        print("Error when saving the code to s3")

def lambda_handler(event,context):
    #TODO implement
    event=json.loads(event['body'])
    blogtopic=event['blog_topic']
    
    genrate_blog=blog_genration_using_bedrock(blogtopic=blogtopic)

    if genrate_blog:
        current_time=datetime.now().strftime(
            '%H:%M:%S'
        )
        s3_key=f"blog_output/{current_time}.txt"
        s3_bucket="aws_bedrock_course1"
        save_blog_details_s3(s3_key,s3_bucket,genrate_blog)

    else:
        print("No blog was genrated")

    return{
        "statusCode":200,
        'body':json.dumps("Blog Genration is completed")
    }

