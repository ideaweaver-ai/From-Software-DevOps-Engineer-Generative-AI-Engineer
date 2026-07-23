import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

response = client.converse(
    modelId="amazon.nova-lite-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "Explain Amazon Bedrock in simple language."
                }
            ]
        }
    ]
)

print(response["output"]["message"]["content"][0]["text"])
