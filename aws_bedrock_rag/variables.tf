# ------------------------------------------------------------------------------
# General
# ------------------------------------------------------------------------------
variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "us-east-1"
}

variable "knowledge_base_name" {
  description = "Name of the Bedrock Knowledge Base"
  type        = string
  default     = "hr-qa-knowledge-base"
}

variable "knowledge_base_description" {
  description = "Description for the knowledge base"
  type        = string
  default     = "HR Q&A Knowledge Base powered by Amazon Bedrock"
}

# ------------------------------------------------------------------------------
# S3 Data Source
# ------------------------------------------------------------------------------
variable "s3_bucket_name" {
  description = "Name of the S3 bucket that stores knowledge base documents"
  type        = string
}

variable "force_destroy_s3" {
  description = "Allow Terraform to destroy the S3 bucket even if it contains objects"
  type        = bool
  default     = false
}

variable "s3_inclusion_prefixes" {
  description = "S3 key prefixes to include when syncing documents (e.g. [\"documents/\"])"
  type        = list(string)
  default     = null
}

# ------------------------------------------------------------------------------
# Embedding Model
# ------------------------------------------------------------------------------
variable "embedding_model_id" {
  description = "Bedrock foundation model ID for generating embeddings"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "embedding_dimension" {
  description = "Dimension of the embedding vectors (must match the chosen model)"
  type        = number
  default     = 1024
}

# ------------------------------------------------------------------------------
# OpenSearch Serverless (Vector Store)
# ------------------------------------------------------------------------------
variable "opensearch_standby_replicas" {
  description = "Enable standby replicas for the OpenSearch Serverless collection (adds cost)"
  type        = bool
  default     = false
}

variable "vector_index_name" {
  description = "Name of the vector index inside the OpenSearch collection"
  type        = string
  default     = "bedrock-kb-index"
}

variable "vector_field_name" {
  description = "Field name for the vector embeddings in the OpenSearch index"
  type        = string
  default     = "embedding"
}

variable "text_field_name" {
  description = "Field name for the raw text chunks in the OpenSearch index"
  type        = string
  default     = "AMAZON_BEDROCK_TEXT_CHUNK"
}

variable "metadata_field_name" {
  description = "Field name for the document metadata in the OpenSearch index"
  type        = string
  default     = "AMAZON_BEDROCK_METADATA"
}

# ------------------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------------------
variable "chunking_strategy" {
  description = "Document chunking strategy: FIXED_SIZE, NONE, or HIERARCHICAL"
  type        = string
  default     = "FIXED_SIZE"

  validation {
    condition     = contains(["FIXED_SIZE", "NONE", "HIERARCHICAL"], var.chunking_strategy)
    error_message = "chunking_strategy must be one of: FIXED_SIZE, NONE, HIERARCHICAL"
  }
}

variable "chunk_max_tokens" {
  description = "Maximum number of tokens per chunk (used with FIXED_SIZE strategy)"
  type        = number
  default     = 300
}

variable "chunk_overlap_percentage" {
  description = "Percentage of overlap between consecutive chunks (used with FIXED_SIZE strategy)"
  type        = number
  default     = 20
}

# ------------------------------------------------------------------------------
# Lambda
# ------------------------------------------------------------------------------
variable "llm_model_id" {
  description = "Bedrock foundation model used for answer generation (RetrieveAndGenerate)"
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

variable "kb_max_results" {
  description = "Number of retrieved passages to send to the LLM for answer generation"
  type        = number
  default     = 5
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60
}

variable "lambda_memory_size" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 256
}

variable "lambda_log_retention_days" {
  description = "CloudWatch log group retention in days"
  type        = number
  default     = 14
}

variable "lambda_log_level" {
  description = "Python logging level for the Lambda function"
  type        = string
  default     = "INFO"
}

# ------------------------------------------------------------------------------
# API Gateway
# ------------------------------------------------------------------------------
variable "create_api_gateway" {
  description = "Whether to create an API Gateway REST API in front of the Lambda"
  type        = bool
  default     = true
}

variable "api_stage_name" {
  description = "API Gateway deployment stage name"
  type        = string
  default     = "v1"
}
