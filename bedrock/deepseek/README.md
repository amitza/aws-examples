# Deepseek inference with Langchain

This Python script provides inference example for Deepseek with AWS Bedrock based on Langchain.

## Requirements
* Python >= 3.11
* AWS account with Bedrock access
* AWS credentials configured locally

## Installation
1. Clone the repository
2. Install the required packages:
    ```shell
    pip install -r requirements.txt
    ```
3. Configure your AWS credentials:
    ```shell
    aws configure
    ```
4. Set the imported model's ARN in the `.env` file.

## Usage
```
python inference.py --prompt "My 10-year-old is refusing to do their homework. Can you suggest some positive reinforcement strategies and creative ways to make homework time more engaging?"
```