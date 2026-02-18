#!/bin/bash
set -e

# Cost guardrails for RallyCut AWS infrastructure.
# Idempotent — safe to re-run.
#
# Usage: ./scripts/setup-aws-guardrails.sh <stage> <email> [budget_amount]
#
# Required IAM permissions: budgets:*, lambda:PutFunctionConcurrency,
#   s3api:PutBucketLifecycleConfiguration, sns:CreateTopic, sns:Subscribe,
#   cloudwatch:PutMetricAlarm

STAGE=${1:?Usage: $0 <stage> <email> [budget_amount]}
EMAIL=${2:?Usage: $0 <stage> <email> [budget_amount]}
BUDGET_AMOUNT=${3:-10}

[[ "$BUDGET_AMOUNT" =~ ^[0-9]+$ ]] || { echo "ERROR: budget_amount must be a positive integer"; exit 1; }
[[ "$EMAIL" =~ ^[^@\"\\]+@[^@\"\\]+\.[^@\"\\]+$ ]] || { echo "ERROR: invalid email address"; exit 1; }

REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>&1) || {
  echo "ERROR: AWS credentials not configured or expired. Run 'aws configure' first."
  exit 1
}
BUCKET_NAME="rallycut-${STAGE}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛡️  RallyCut Cost Guardrails — stage: ${STAGE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ─── Section A: AWS Budget ────────────────────────────────────

BUDGET_NAME="rallycut-${STAGE}-monthly"

echo "📊 Setting up AWS Budget: \$${BUDGET_AMOUNT}/month..."

# Budget API requires us-east-1 regardless of configured region
# Delete existing budget for idempotency (no upsert in budget API)
aws budgets delete-budget \
  --account-id "$ACCOUNT_ID" \
  --budget-name "$BUDGET_NAME" \
  --region us-east-1 2>/dev/null || true

aws budgets create-budget \
  --account-id "$ACCOUNT_ID" \
  --region us-east-1 \
  --budget '{
    "BudgetName": "'"$BUDGET_NAME"'",
    "BudgetLimit": {
      "Amount": "'"$BUDGET_AMOUNT"'",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[
    {
      "Notification": {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 50,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "'"$EMAIL"'"}]
    },
    {
      "Notification": {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 80,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "'"$EMAIL"'"}]
    },
    {
      "Notification": {
        "NotificationType": "FORECASTED",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 100,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "'"$EMAIL"'"}]
    }
  ]'

echo "   ✅ Budget: \$${BUDGET_AMOUNT}/month with alerts at 50%, 80%, 100% forecasted"

# ─── Section B: Lambda Concurrency Limits ─────────────────────

echo ""
echo "⚡ Setting Lambda concurrency limits..."

EXPORT_FN="rallycut-video-export-${STAGE}"
OPTIMIZE_FN="rallycut-video-optimize-${STAGE}"

# put-function-concurrency is inherently idempotent
# Accounts with default concurrency limit (10) cannot reserve — requires 10 unreserved minimum
set_concurrency() {
  local fn_name=$1 limit=$2
  if ! aws lambda get-function --function-name "$fn_name" --region "$REGION" >/dev/null 2>&1; then
    echo "   ⏭️  ${fn_name}: function not found, skipping"
  elif aws lambda put-function-concurrency \
    --function-name "$fn_name" \
    --reserved-concurrent-executions "$limit" \
    --region "$REGION" >/dev/null 2>&1; then
    echo "   ✅ ${fn_name}: reserved concurrency = ${limit}"
  else
    echo "   ⚠️  ${fn_name}: could not set concurrency (account limit too low, request increase via AWS Support)"
  fi
}
set_concurrency "$EXPORT_FN" 2
set_concurrency "$OPTIMIZE_FN" 3

# ─── Section C: S3 Lifecycle Policy ──────────────────────────

echo ""
echo "🗄️  Setting S3 lifecycle policy on ${BUCKET_NAME}..."

if ! aws s3api head-bucket --bucket "$BUCKET_NAME" --region "$REGION" 2>/dev/null; then
  echo "   ⏭️  Bucket ${BUCKET_NAME} not found, skipping lifecycle policy"
else

aws s3api put-bucket-lifecycle-configuration \
  --bucket "$BUCKET_NAME" \
  --region "$REGION" \
  --lifecycle-configuration '{
  "Rules": [
    {
      "ID": "ExportsCleanup",
      "Status": "Enabled",
      "Filter": {"Prefix": "exports/"},
      "Expiration": {"Days": 30}
    },
    {
      "ID": "VideosTiering",
      "Status": "Enabled",
      "Filter": {"Prefix": "videos/"},
      "Transitions": [
        {"Days": 30, "StorageClass": "INTELLIGENT_TIERING"}
      ]
    },
    {
      "ID": "AnalysisTiering",
      "Status": "Enabled",
      "Filter": {"Prefix": "analysis/"},
      "Transitions": [
        {"Days": 30, "StorageClass": "STANDARD_IA"}
      ]
    },
    {
      "ID": "AbortIncompleteUploads",
      "Status": "Enabled",
      "Filter": {"Prefix": ""},
      "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 3}
    }
  ]
}'

echo "   ✅ exports/: delete at 30d"
echo "   ✅ videos/: Intelligent-Tiering at 30d (app manages retention)"
echo "   ✅ analysis/: Standard-IA at 30d"
echo "   ✅ Abort incomplete multipart uploads after 3d"

fi  # bucket exists

# ─── Section D: CloudWatch Alarms ────────────────────────────

echo ""
echo "🔔 Setting up CloudWatch alarms..."

SNS_TOPIC="rallycut-${STAGE}-lambda-alerts"

TOPIC_ARN=$(aws sns create-topic \
  --name "$SNS_TOPIC" \
  --region "$REGION" \
  --query 'TopicArn' --output text)

aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint "$EMAIL" \
  --region "$REGION" >/dev/null || true

echo "   ✅ SNS topic: ${SNS_TOPIC}"

# Export Lambda: timeout=900s, alert at p99 > 720s (80%)
# Optimize Lambda: timeout=600s, alert at p99 > 480s (80%)
for FN_NAME in "$EXPORT_FN" "$OPTIMIZE_FN"; do
  case "$FN_NAME" in
    "$EXPORT_FN")  THRESHOLD_MS=720000 ;;
    "$OPTIMIZE_FN") THRESHOLD_MS=480000 ;;
  esac

  if ! aws lambda get-function --function-name "$FN_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "   ⏭️  ${FN_NAME}: function not found, skipping alarms"
    continue
  fi

  # Duration alarm: p99 > 80% of timeout
  aws cloudwatch put-metric-alarm \
    --alarm-name "${FN_NAME}-duration" \
    --alarm-description "p99 duration approaching timeout for ${FN_NAME}" \
    --namespace AWS/Lambda \
    --metric-name Duration \
    --dimensions "Name=FunctionName,Value=${FN_NAME}" \
    --extended-statistic p99 \
    --period 300 \
    --evaluation-periods 1 \
    --threshold "$THRESHOLD_MS" \
    --comparison-operator GreaterThanThreshold \
    --alarm-actions "$TOPIC_ARN" \
    --treat-missing-data notBreaching \
    --region "$REGION" >/dev/null

  # Error alarm: >3 errors in 5 minutes
  aws cloudwatch put-metric-alarm \
    --alarm-name "${FN_NAME}-errors" \
    --alarm-description "High error rate for ${FN_NAME}" \
    --namespace AWS/Lambda \
    --metric-name Errors \
    --dimensions "Name=FunctionName,Value=${FN_NAME}" \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 3 \
    --comparison-operator GreaterThanThreshold \
    --alarm-actions "$TOPIC_ARN" \
    --treat-missing-data notBreaching \
    --region "$REGION" >/dev/null

  echo "   ✅ ${FN_NAME}: duration + error alarms"
done

# ─── Section E: Summary ──────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛡️  Guardrails configured for stage: ${STAGE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Budget:     \$${BUDGET_AMOUNT}/month (alerts → ${EMAIL})"
echo "  Lambda:     export=2, optimize=3 concurrent"
echo "  S3:         prefix-based lifecycle (no blanket deletion)"
echo "  Alarms:     duration + error alerts → ${EMAIL}"
echo ""
echo "⚠️  Action required:"
echo "  1. Check ${EMAIL} for SNS subscription confirmation email"
echo "  2. Click the confirmation link to activate alarm notifications"
echo ""
echo "📋 Modal spending limit (manual):"
echo "  1. Visit https://modal.com/settings → Usage/Billing"
echo "  2. Set spending limit to \$10/month"
echo "  3. Verify: modal profile list"
echo ""
echo "🔍 Verify:"
echo "  aws budgets describe-budget --account-id ${ACCOUNT_ID} --budget-name ${BUDGET_NAME} --region us-east-1"
echo "  aws s3api get-bucket-lifecycle-configuration --bucket ${BUCKET_NAME}"
echo "  aws cloudwatch describe-alarms --alarm-name-prefix rallycut-video"
