terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.50.0"
    }
    opensearch = {
      source  = "opensearch-project/opensearch"
      version = ">= 2.3.0"
    }
    time = {
      source  = "hashicorp/time"
      version = ">= 0.9.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

# ------------------------------------------------------------------------------
# S3 Bucket — Data source for the Knowledge Base
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "kb_data_source" {
  bucket        = var.s3_bucket_name
  force_destroy = var.force_destroy_s3
}

resource "aws_s3_bucket_server_side_encryption_configuration" "kb_data_source" {
  bucket = aws_s3_bucket.kb_data_source.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "kb_data_source" {
  bucket = aws_s3_bucket.kb_data_source.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "kb_data_source" {
  bucket = aws_s3_bucket.kb_data_source.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ------------------------------------------------------------------------------
# OpenSearch Serverless — Vector store for embeddings
# ------------------------------------------------------------------------------
resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${var.knowledge_base_name}-enc"
  type = "encryption"
  policy = jsonencode({
    Rules = [
      {
        Resource     = ["collection/${var.knowledge_base_name}-vectors"]
        ResourceType = "collection"
      }
    ]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "network" {
  name = "${var.knowledge_base_name}-net"
  type = "network"
  policy = jsonencode([
    {
      Rules = [
        {
          Resource     = ["collection/${var.knowledge_base_name}-vectors"]
          ResourceType = "collection"
        }
      ]
      AllowFromPublic = true
    }
  ])
}

resource "aws_opensearchserverless_collection" "kb_vectors" {
  name             = "${var.knowledge_base_name}-vectors"
  type             = "VECTORSEARCH"
  standby_replicas = var.opensearch_standby_replicas ? "ENABLED" : "DISABLED"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
  ]
}

resource "aws_opensearchserverless_access_policy" "kb_access" {
  name = "${var.knowledge_base_name}-access"
  type = "data"
  policy = jsonencode([
    {
      Rules = [
        {
          Resource     = ["collection/${var.knowledge_base_name}-vectors"]
          ResourceType = "collection"
          Permission = [
            "aoss:CreateCollectionItems",
            "aoss:DeleteCollectionItems",
            "aoss:UpdateCollectionItems",
            "aoss:DescribeCollectionItems",
          ]
        },
        {
          Resource     = ["index/${var.knowledge_base_name}-vectors/*"]
          ResourceType = "index"
          Permission = [
            "aoss:CreateIndex",
            "aoss:DeleteIndex",
            "aoss:UpdateIndex",
            "aoss:DescribeIndex",
            "aoss:ReadDocument",
            "aoss:WriteDocument",
          ]
        }
      ]
      Principal = [
        aws_iam_role.bedrock_kb_role.arn,
        data.aws_caller_identity.current.arn,
      ]
    }
  ])
}

# Wait for the collection to become active before creating the index
resource "time_sleep" "wait_for_collection" {
  depends_on      = [aws_opensearchserverless_collection.kb_vectors]
  create_duration = "60s"
}

# Configure the OpenSearch provider to talk to the serverless collection
provider "opensearch" {
  url         = aws_opensearchserverless_collection.kb_vectors.collection_endpoint
  healthcheck = false
}

resource "opensearch_index" "kb_vector_index" {
  name                           = var.vector_index_name
  number_of_shards               = "2"
  number_of_replicas             = "0"
  index_knn                      = true
  index_knn_algo_param_ef_search = "512"
  force_destroy                  = true

  mappings = jsonencode({
    properties = {
      "${var.vector_field_name}" = {
        type      = "knn_vector"
        dimension = var.embedding_dimension
        method = {
          name       = "hnsw"
          engine     = "faiss"
          parameters = {}
        }
      }
      "${var.text_field_name}" = {
        type  = "text"
        index = true
      }
      "${var.metadata_field_name}" = {
        type  = "text"
        index = false
      }
    }
  })

  depends_on = [
    time_sleep.wait_for_collection,
    aws_opensearchserverless_access_policy.kb_access,
  ]
}

# ------------------------------------------------------------------------------
# IAM Role — Bedrock Knowledge Base execution role
# ------------------------------------------------------------------------------
resource "aws_iam_role" "bedrock_kb_role" {
  name = "${var.knowledge_base_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = local.account_id
          }
          ArnLike = {
            "aws:SourceArn" = "arn:aws:bedrock:${local.region}:${local.account_id}:knowledge-base/*"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "bedrock_kb_model" {
  name = "BedrockModelAccess"
  role = aws_iam_role.bedrock_kb_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "bedrock:InvokeModel"
        Resource = "arn:aws:bedrock:${local.region}::foundation-model/${var.embedding_model_id}"
      }
    ]
  })
}

resource "aws_iam_role_policy" "bedrock_kb_s3" {
  name = "S3DataSourceAccess"
  role = aws_iam_role.bedrock_kb_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.kb_data_source.arn,
          "${aws_s3_bucket.kb_data_source.arn}/*",
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "bedrock_kb_opensearch" {
  name = "OpenSearchServerlessAccess"
  role = aws_iam_role.bedrock_kb_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "aoss:APIAccessAll"
        Resource = aws_opensearchserverless_collection.kb_vectors.arn
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# Bedrock Knowledge Base
# ------------------------------------------------------------------------------
resource "aws_bedrockagent_knowledge_base" "this" {
  name     = var.knowledge_base_name
  role_arn = aws_iam_role.bedrock_kb_role.arn

  description = var.knowledge_base_description

  knowledge_base_configuration {
    type = "VECTOR"

    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${local.region}::foundation-model/${var.embedding_model_id}"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"

    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.kb_vectors.arn
      vector_index_name = var.vector_index_name

      field_mapping {
        vector_field   = var.vector_field_name
        text_field     = var.text_field_name
        metadata_field = var.metadata_field_name
      }
    }
  }

  depends_on = [
    aws_iam_role_policy.bedrock_kb_model,
    aws_iam_role_policy.bedrock_kb_s3,
    aws_iam_role_policy.bedrock_kb_opensearch,
    opensearch_index.kb_vector_index,
  ]
}

# ------------------------------------------------------------------------------
# Knowledge Base Data Source (S3)
# ------------------------------------------------------------------------------
resource "aws_bedrockagent_data_source" "s3" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.this.id
  name              = "${var.knowledge_base_name}-s3"

  data_source_configuration {
    type = "S3"

    s3_configuration {
      bucket_arn              = aws_s3_bucket.kb_data_source.arn
      inclusion_prefixes      = var.s3_inclusion_prefixes
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = var.chunking_strategy

      dynamic "fixed_size_chunking_configuration" {
        for_each = var.chunking_strategy == "FIXED_SIZE" ? [1] : []
        content {
          max_tokens         = var.chunk_max_tokens
          overlap_percentage = var.chunk_overlap_percentage
        }
      }
    }
  }
}

# ------------------------------------------------------------------------------
# Lambda Function — Query the Knowledge Base
# ------------------------------------------------------------------------------
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/lambda"
  output_path = "${path.module}/.build/lambda.zip"
}

resource "aws_iam_role" "lambda_role" {
  name = "${var.knowledge_base_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_bedrock" {
  name = "BedrockKBQueryAccess"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:RetrieveAndGenerate",
          "bedrock:Retrieve",
        ]
        Resource = aws_bedrockagent_knowledge_base.this.arn
      },
      {
        Effect = "Allow"
        Action = "bedrock:InvokeModel"
        Resource = "arn:aws:bedrock:${local.region}::foundation-model/${var.llm_model_id}"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${var.knowledge_base_name}-query"
  retention_in_days = var.lambda_log_retention_days
}

resource "aws_lambda_function" "kb_query" {
  function_name    = "${var.knowledge_base_name}-query"
  role             = aws_iam_role.lambda_role.arn
  handler          = "index.handler"
  runtime          = "python3.12"
  timeout          = var.lambda_timeout
  memory_size      = var.lambda_memory_size
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      KNOWLEDGE_BASE_ID = aws_bedrockagent_knowledge_base.this.id
      MODEL_ARN         = "arn:aws:bedrock:${local.region}::foundation-model/${var.llm_model_id}"
      MAX_RESULTS       = tostring(var.kb_max_results)
      LOG_LEVEL         = var.lambda_log_level
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda_logs,
    aws_iam_role_policy.lambda_bedrock,
  ]
}

# ------------------------------------------------------------------------------
# API Gateway (REST) — HTTP endpoint for the Lambda
# ------------------------------------------------------------------------------
resource "aws_api_gateway_rest_api" "kb_api" {
  count = var.create_api_gateway ? 1 : 0

  name        = "${var.knowledge_base_name}-api"
  description = "REST API for querying the ${var.knowledge_base_name} Knowledge Base"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

resource "aws_api_gateway_resource" "query" {
  count = var.create_api_gateway ? 1 : 0

  rest_api_id = aws_api_gateway_rest_api.kb_api[0].id
  parent_id   = aws_api_gateway_rest_api.kb_api[0].root_resource_id
  path_part   = "query"
}

resource "aws_api_gateway_method" "post_query" {
  count = var.create_api_gateway ? 1 : 0

  rest_api_id   = aws_api_gateway_rest_api.kb_api[0].id
  resource_id   = aws_api_gateway_resource.query[0].id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  count = var.create_api_gateway ? 1 : 0

  rest_api_id             = aws_api_gateway_rest_api.kb_api[0].id
  resource_id             = aws_api_gateway_resource.query[0].id
  http_method             = aws_api_gateway_method.post_query[0].http_method
  type                    = "AWS_PROXY"
  integration_http_method = "POST"
  uri                     = aws_lambda_function.kb_query.invoke_arn
}

# CORS: OPTIONS method
resource "aws_api_gateway_method" "options_query" {
  count = var.create_api_gateway ? 1 : 0

  rest_api_id   = aws_api_gateway_rest_api.kb_api[0].id
  resource_id   = aws_api_gateway_resource.query[0].id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "options_integration" {
  count = var.create_api_gateway ? 1 : 0

  rest_api_id = aws_api_gateway_rest_api.kb_api[0].id
  resource_id = aws_api_gateway_resource.query[0].id
  http_method = aws_api_gateway_method.options_query[0].http_method
  type        = "MOCK"

  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}

resource "aws_api_gateway_method_response" "options_200" {
  count = var.create_api_gateway ? 1 : 0

  rest_api_id = aws_api_gateway_rest_api.kb_api[0].id
  resource_id = aws_api_gateway_resource.query[0].id
  http_method = aws_api_gateway_method.options_query[0].http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }

  response_models = {
    "application/json" = "Empty"
  }
}

resource "aws_api_gateway_integration_response" "options_response" {
  count = var.create_api_gateway ? 1 : 0

  rest_api_id = aws_api_gateway_rest_api.kb_api[0].id
  resource_id = aws_api_gateway_resource.query[0].id
  http_method = aws_api_gateway_method.options_query[0].http_method
  status_code = aws_api_gateway_method_response.options_200[0].status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type'"
    "method.response.header.Access-Control-Allow-Methods" = "'POST,OPTIONS'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }

  depends_on = [aws_api_gateway_integration.options_integration]
}

resource "aws_api_gateway_deployment" "kb_api" {
  count = var.create_api_gateway ? 1 : 0

  rest_api_id = aws_api_gateway_rest_api.kb_api[0].id

  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.query[0].id,
      aws_api_gateway_method.post_query[0].id,
      aws_api_gateway_integration.lambda_integration[0].id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_api_gateway_stage" "kb_api" {
  count = var.create_api_gateway ? 1 : 0

  deployment_id = aws_api_gateway_deployment.kb_api[0].id
  rest_api_id   = aws_api_gateway_rest_api.kb_api[0].id
  stage_name    = var.api_stage_name
}

resource "aws_lambda_permission" "api_gateway" {
  count = var.create_api_gateway ? 1 : 0

  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.kb_query.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.kb_api[0].execution_arn}/*/*"
}
