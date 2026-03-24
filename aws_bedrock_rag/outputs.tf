output "knowledge_base_id" {
  description = "ID of the Bedrock Knowledge Base"
  value       = aws_bedrockagent_knowledge_base.this.id
}

output "knowledge_base_arn" {
  description = "ARN of the Bedrock Knowledge Base"
  value       = aws_bedrockagent_knowledge_base.this.arn
}

output "data_source_id" {
  description = "ID of the S3 data source attached to the Knowledge Base"
  value       = aws_bedrockagent_data_source.s3.data_source_id
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for knowledge base documents"
  value       = aws_s3_bucket.kb_data_source.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.kb_data_source.arn
}

output "opensearch_collection_arn" {
  description = "ARN of the OpenSearch Serverless collection"
  value       = aws_opensearchserverless_collection.kb_vectors.arn
}

output "opensearch_collection_endpoint" {
  description = "Endpoint of the OpenSearch Serverless collection"
  value       = aws_opensearchserverless_collection.kb_vectors.collection_endpoint
}

output "bedrock_kb_role_arn" {
  description = "ARN of the IAM role used by the Knowledge Base"
  value       = aws_iam_role.bedrock_kb_role.arn
}

# Lambda
output "lambda_function_name" {
  description = "Name of the Lambda function that queries the Knowledge Base"
  value       = aws_lambda_function.kb_query.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.kb_query.arn
}

# API Gateway
output "api_gateway_url" {
  description = "Invoke URL of the API Gateway endpoint (POST /query)"
  value       = var.create_api_gateway ? "${aws_api_gateway_stage.kb_api[0].invoke_url}/query" : null
}
