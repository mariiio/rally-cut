#!/bin/bash
set -e

STAGE=${1:-dev}
REGION=${AWS_REGION:-us-east-1}
FUNCTION_NAME="rallycut-video-export-${STAGE}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/rallycut-video-export-${STAGE}:latest"
ROLE_NAME="${FUNCTION_NAME}-role"
S3_BUCKET="rallycut-${STAGE}"

echo "🚀 Deploying Lambda function: $FUNCTION_NAME"

# Create IAM role if it doesn't exist
echo "📋 Setting up IAM role..."
if ! aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
  aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'
  
  # Attach basic Lambda execution policy
  aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  
  # Add S3 permissions
  aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name S3Access \
    --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [{
        \"Effect\": \"Allow\",
        \"Action\": [\"s3:GetObject\", \"s3:PutObject\"],
        \"Resource\": \"arn:aws:s3:::${S3_BUCKET}/*\"
      }]
    }"
  
  echo "⏳ Waiting for role to propagate..."
  sleep 10
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Create or update Lambda function
echo "🔧 Creating/updating Lambda function..."
if aws lambda get-function --function-name $FUNCTION_NAME 2>/dev/null; then
  aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --image-uri $ECR_IMAGE
else
  aws lambda create-function \
    --function-name $FUNCTION_NAME \
    --package-type Image \
    --code ImageUri=$ECR_IMAGE \
    --role $ROLE_ARN \
    --timeout 900 \
    --memory-size 1024 \
    --architectures arm64 \
    --environment "Variables={S3_BUCKET_NAME=${S3_BUCKET}}"
fi

echo ""
echo "✅ Lambda deployed: $FUNCTION_NAME"
echo ""
echo "Add to your API .env:"
echo "  EXPORT_LAMBDA_FUNCTION_NAME=$FUNCTION_NAME"
