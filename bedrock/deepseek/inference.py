from dotenv import load_dotenv
from langchain_aws import BedrockLLM
from transformers import AutoTokenizer, PreTrainedTokenizer
import boto3
from botocore.config import Config
import os
import argparse
import sys

load_dotenv()

def setup_bedrock_client(region: str) -> boto3.client:
    """Sets up and configures an AWS Bedrock client.

    Initializes a boto3 session and creates a Bedrock client with specified
    configuration parameters for timeouts and retries.

    Args:
        region: AWS region identifier.

    Returns:
        A configured boto3 client for Bedrock runtime.
    """
    session = boto3.Session()
    return session.client(
        service_name='bedrock-runtime',
        region_name=region,
        config=Config(
            connect_timeout=60,
            read_timeout=60,
            retries={'max_attempts': 5}
        )
    )

def setup_bedrock_llm(region: str, imported_model_id: str, max_tokens: int=4096, top_p: float=0.9, temperature: float= 0.3) -> BedrockLLM:
    """Creates and configures a BedrockLLM instance.

    Sets up a BedrockLLM with specified parameters for text generation and
    model configuration.

    Args:
        region: AWS region identifier.
        max_tokens: Maximum number of tokens.
        top_p: Top-p sampling parameter.
        temperature: Temperature parameter.

    Returns:
        A configured BedrockLLM instance ready for inference.
    """
    model_kwargs = { 
        "max_tokens": max_tokens,
        "top_p": top_p
        }
    
    model = BedrockLLM(
        client=setup_bedrock_client(region=region),
        model=imported_model_id,
        model_kwargs=model_kwargs,
        provider='meta',
        temperature=temperature
    )

    return model

def invoke_model(model: BedrockLLM, tokenizer: PreTrainedTokenizer, raw_prompt: str) -> str:
    """Sends a prompt to the model and returns its response.

    Processes the input prompt through the tokenizer and sends it to the model
    for inference.

    Args:
        model: Configured BedrockLLM instance.
        tokenizer: HuggingFace tokenizer for processing the prompt.
        raw_prompt: The text prompt to send to the model.

    Returns:
        The model's response as a string.
    """
    messages = [{"role": "user", "content": raw_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return model.invoke(prompt)

if __name__ == "__main__":
    region_info = os.getenv('AWS_REGION', 'us-west-2')
    model_id = os.getenv('MODEL_ID', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    imported_model_id = os.getenv('IMPORTED_MODEL_ID')

    parser = argparse.ArgumentParser(
        description='Deepseek inference using AWS Bedrock'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='The prompt text to send to the model',
        required=True
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        print("\nError: The --prompt argument is required.")
        print("Usage: python script.py --prompt \"Your prompt text here\"")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bedrock_runtime_client = setup_bedrock_client(region=region_info)
    model = setup_bedrock_llm(region=region_info, imported_model_id=imported_model_id)

    response = invoke_model(model=model, tokenizer=tokenizer, raw_prompt=args.prompt)

    print(f'Model response: {response}')