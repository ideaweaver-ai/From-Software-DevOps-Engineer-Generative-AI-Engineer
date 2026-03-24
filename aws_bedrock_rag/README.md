# HR Q&A — Bedrock Knowledge Base with Terraform

An end-to-end RAG (Retrieval-Augmented Generation) solution that lets employees ask HR policy questions in natural language and receive answers grounded in company documents. Infrastructure is fully managed through Terraform; the frontend is a Streamlit web app.

---

## Architecture

```
┌──────────────┐      HTTPS POST       ┌───────────────┐      invoke       ┌────────────────┐
│   Streamlit  │ ───────────────────▶  │  API Gateway  │ ───────────────▶ │    Lambda fn   │
│   Frontend   │ ◀─────────────────── │  (REST, /query)│ ◀─────────────── │  (Python 3.12) │
└──────────────┘      JSON response    └───────────────┘                  └───────┬────────┘
                                                                                  │
                                                                  RetrieveAndGenerate
                                                                                  │
                                                                          ┌───────▼────────┐
                                                                          │ Bedrock KB     │
                                                                          │ (Knowledge Base)│
                                                                          └───────┬────────┘
                                                                                  │
                                                        ┌─────────────────────────┼──────────────────────────┐
                                                        │                         │                          │
                                                ┌───────▼───────┐   ┌─────────────▼──────────┐   ┌──────────▼─────────┐
                                                │  S3 Bucket    │   │  OpenSearch Serverless  │   │  Bedrock Models    │
                                                │  (HR docs)    │   │  (vector store)         │   │  Titan Embed + LLM │
                                                └───────────────┘   └────────────────────────┘   └────────────────────┘
```

**Flow:**
1. User types an HR question in the Streamlit UI.
2. The frontend POSTs the question to API Gateway.
3. API Gateway proxies the request to the Lambda function.
4. Lambda calls the Bedrock `RetrieveAndGenerate` API with the Knowledge Base ID.
5. Bedrock retrieves relevant document chunks from OpenSearch Serverless and generates an answer using the configured LLM (Claude 3 Haiku by default).
6. Lambda returns the answer + source citations back through API Gateway to the frontend.

---

## Project Structure

```
.
├── main.tf                  # All AWS resources (S3, OpenSearch, IAM, Bedrock KB, Lambda, API GW)
├── variables.tf             # Input variables with defaults
├── outputs.tf               # Useful output values (IDs, ARNs, URLs)
├── terraform.tfvars.example # Sample variable values — copy and customise
├── lambda/
│   └── index.py             # Lambda function code (Bedrock RetrieveAndGenerate)
├── rag_frontend.py          # Streamlit frontend
├── rag_backend.py           # (Legacy) Local LangChain backend — not used in this setup
├── data_load_test.py        # PDF loader test script
├── data_split_test.py       # Text splitter test script
└── README.md                # This file
```

---

## Prerequisites

Before you begin, make sure you have:

- **Terraform >= 1.5** installed — [Install guide](https://developer.hashicorp.com/terraform/install)
- **AWS CLI v2** configured with credentials that have admin or sufficient IAM permissions
- **Python >= 3.10** (for running the Streamlit frontend locally)
- **Bedrock model access enabled** in your AWS account for:
  - `amazon.titan-embed-text-v2:0` (embeddings)
  - `anthropic.claude-3-haiku-20240307-v1:0` (answer generation)
  - Go to [Amazon Bedrock → Model access](https://console.aws.amazon.com/bedrock/home#/modelaccess) in the AWS Console and request access if not already enabled.

---

## Step-by-Step Deployment

### 1. Clone and Configure Variables

```bash
cd HR_QA_26June2025

# Create your variables file from the example
cp terraform.tfvars.example terraform.tfvars
```

Open `terraform.tfvars` and update the values:

```hcl
# REQUIRED — must be globally unique
s3_bucket_name = "your-company-hr-qa-docs-unique-name"

# OPTIONAL — change these if needed
aws_region          = "us-east-1"
knowledge_base_name = "hr-qa-knowledge-base"
llm_model_id        = "anthropic.claude-3-haiku-20240307-v1:0"
```

> **Important:** The `s3_bucket_name` must be globally unique across all AWS accounts. Pick something specific to your organisation.

### 2. Initialise Terraform

```bash
terraform init
```

This downloads the required providers:
- `hashicorp/aws` (>= 5.50.0)
- `opensearch-project/opensearch` (>= 2.3.0)
- `hashicorp/time` (>= 0.9.0)

### 3. Review the Plan

```bash
terraform plan
```

You should see approximately **20+ resources** to be created:
- 1 S3 bucket (with encryption, versioning, public access block)
- 4 OpenSearch Serverless resources (collection, encryption policy, network policy, access policy)
- 1 OpenSearch vector index
- 3 IAM roles/policies for Bedrock KB
- 1 Bedrock Knowledge Base
- 1 Bedrock Data Source
- 1 Lambda function (with IAM role, policies, CloudWatch log group)
- 7 API Gateway resources (REST API, resource, methods, integration, deployment, stage, permission)

### 4. Deploy

```bash
terraform apply
```

Type `yes` when prompted. Deployment takes **3-5 minutes** (the OpenSearch Serverless collection needs ~60s to become active).

After completion, Terraform prints the outputs:

```
api_gateway_url            = "https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/v1/query"
knowledge_base_id          = "XXXXXXXXXX"
data_source_id             = "XXXXXXXXXX"
lambda_function_name       = "hr-qa-knowledge-base-query"
s3_bucket_name             = "your-company-hr-qa-docs-unique-name"
opensearch_collection_endpoint = "https://xxxxxxxxxx.us-east-1.aoss.amazonaws.com"
```

### 5. Upload Your HR Documents to S3

Upload the PDF/text/HTML documents that the Knowledge Base should use as its source:

```bash
# Upload a single file
aws s3 cp ./Leave-Policy.pdf s3://$(terraform output -raw s3_bucket_name)/

# Or upload an entire folder
aws s3 cp ./hr-documents/ s3://$(terraform output -raw s3_bucket_name)/documents/ --recursive
```

**Supported file types:** PDF, TXT, HTML, Markdown, CSV, DOC/DOCX, XLS/XLSX.

### 6. Sync the Knowledge Base (Start Ingestion)

After uploading documents, trigger an ingestion job so Bedrock processes, chunks, embeds, and indexes them:

```bash
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id "$(terraform output -raw knowledge_base_id)" \
  --data-source-id "$(terraform output -raw data_source_id)"
```

Monitor the job status:

```bash
aws bedrock-agent list-ingestion-jobs \
  --knowledge-base-id "$(terraform output -raw knowledge_base_id)" \
  --data-source-id "$(terraform output -raw data_source_id)"
```

Wait until the status shows `COMPLETE` before testing queries.

> **Note:** Re-run the ingestion job every time you add, update, or remove documents from S3.

### 7. Test the API Endpoint

```bash
curl -X POST "$(terraform output -raw api_gateway_url)" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the company leave policy?"}'
```

Expected response:

```json
{
  "question": "What is the company leave policy?",
  "answer": "According to the HR handbook, employees are entitled to ...",
  "citations": [
    {
      "text": "Section 4.2 — Annual leave entitlement is ...",
      "source": "s3://your-bucket/Leave-Policy.pdf"
    }
  ]
}
```

### 8. Run the Streamlit Frontend

Install the Python dependencies:

```bash
pip install streamlit requests
```

Create the Streamlit secrets file with your API URL:

```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
API_GATEWAY_URL = "PASTE_YOUR_API_GATEWAY_URL_HERE"
EOF
```

Replace the placeholder with the actual URL:

```bash
# macOS / Linux one-liner
echo "API_GATEWAY_URL = \"$(terraform output -raw api_gateway_url)\"" > .streamlit/secrets.toml
```

Launch the app:

```bash
streamlit run rag_frontend.py
```

The app opens at `http://localhost:8501`. Type an HR question and hit **Ask the HR Assistant**.

---

## Terraform Variables Reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `aws_region` | string | `us-east-1` | AWS region for all resources |
| `knowledge_base_name` | string | `hr-qa-knowledge-base` | Name prefix for all resources |
| `knowledge_base_description` | string | `HR Q&A Knowledge Base...` | Description stored in Bedrock |
| `s3_bucket_name` | string | **required** | Globally unique S3 bucket name |
| `force_destroy_s3` | bool | `false` | Allow `terraform destroy` to delete non-empty bucket |
| `s3_inclusion_prefixes` | list(string) | `null` | Limit ingestion to specific S3 prefixes |
| `embedding_model_id` | string | `amazon.titan-embed-text-v2:0` | Bedrock model for embeddings |
| `embedding_dimension` | number | `1024` | Must match the embedding model's output dimension |
| `opensearch_standby_replicas` | bool | `false` | Enable HA replicas (increases cost) |
| `vector_index_name` | string | `bedrock-kb-index` | OpenSearch index name |
| `chunking_strategy` | string | `FIXED_SIZE` | `FIXED_SIZE`, `NONE`, or `HIERARCHICAL` |
| `chunk_max_tokens` | number | `300` | Max tokens per chunk (FIXED_SIZE only) |
| `chunk_overlap_percentage` | number | `20` | Overlap % between chunks (FIXED_SIZE only) |
| `llm_model_id` | string | `anthropic.claude-3-haiku-20240307-v1:0` | LLM for answer generation |
| `kb_max_results` | number | `5` | Number of passages retrieved per query |
| `lambda_timeout` | number | `60` | Lambda timeout in seconds |
| `lambda_memory_size` | number | `256` | Lambda memory in MB |
| `lambda_log_retention_days` | number | `14` | CloudWatch log retention |
| `lambda_log_level` | string | `INFO` | Python log level |
| `create_api_gateway` | bool | `true` | Set `false` to skip API Gateway creation |
| `api_stage_name` | string | `v1` | API Gateway stage name |

---

## Terraform Outputs Reference

| Output | Description |
|---|---|
| `knowledge_base_id` | Bedrock Knowledge Base ID |
| `knowledge_base_arn` | Bedrock Knowledge Base ARN |
| `data_source_id` | S3 data source ID (needed for ingestion jobs) |
| `s3_bucket_name` | Name of the created S3 bucket |
| `s3_bucket_arn` | ARN of the S3 bucket |
| `opensearch_collection_arn` | OpenSearch Serverless collection ARN |
| `opensearch_collection_endpoint` | OpenSearch Serverless endpoint URL |
| `bedrock_kb_role_arn` | IAM role ARN used by the Knowledge Base |
| `lambda_function_name` | Lambda function name |
| `lambda_function_arn` | Lambda function ARN |
| `api_gateway_url` | Full invoke URL for `POST /query` |

---

## Customisation

### Use a Different LLM

Change `llm_model_id` in `terraform.tfvars` to any Bedrock model you have access to:

```hcl
# Claude 3 Sonnet (more capable, higher cost)
llm_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Amazon Titan Text
llm_model_id = "amazon.titan-text-express-v1"
```

Then run `terraform apply` to update the Lambda environment variable.

### Use a Different Embedding Model

If you switch embedding models, you must also update `embedding_dimension` and **recreate** the OpenSearch index + re-ingest documents:

```hcl
# Cohere Embed English v3
embedding_model_id  = "cohere.embed-english-v3"
embedding_dimension = 1024
```

### Restrict S3 Source Folder

To only ingest documents from a specific prefix:

```hcl
s3_inclusion_prefixes = ["policies/", "handbooks/"]
```

### Disable API Gateway

If you only need direct Lambda invocation (e.g., from another AWS service):

```hcl
create_api_gateway = false
```

---

## Cleanup

To destroy all resources:

```bash
# If the S3 bucket contains files, empty it first
aws s3 rm s3://$(terraform output -raw s3_bucket_name) --recursive

# Destroy all Terraform-managed resources
terraform destroy
```

Or set `force_destroy_s3 = true` in your tfvars before destroying, and Terraform will empty the bucket automatically.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `terraform apply` fails on OpenSearch index | The collection may not be ready. Run `terraform apply` again — the 60s sleep usually handles this, but occasionally it needs more time. |
| Lambda returns `ValidationException` | Verify Bedrock model access is enabled in your account for both the embedding and LLM models. |
| Empty answers from the Knowledge Base | Make sure the ingestion job completed successfully (`COMPLETE` status). Check CloudWatch logs at `/aws/lambda/hr-qa-knowledge-base-query`. |
| `AccessDeniedException` on Lambda | Ensure the Lambda IAM role has `bedrock:RetrieveAndGenerate`, `bedrock:Retrieve`, and `bedrock:InvokeModel` permissions. These are set automatically by Terraform. |
| Streamlit shows "Could not reach the API" | Verify the `API_GATEWAY_URL` in `.streamlit/secrets.toml` matches the `api_gateway_url` Terraform output exactly. |
| CORS errors in browser | The API Gateway OPTIONS handler and Lambda response headers both set `Access-Control-Allow-Origin: *`. If you see CORS issues, ensure the API was deployed (check the stage exists in the AWS Console). |

---

## Cost Considerations

| Service | Pricing Notes |
|---|---|
| **OpenSearch Serverless** | Billed by OCU-hours. Minimum 2 OCUs for indexing + 2 OCUs for search = ~$0.24/hr each = **~$350/month** at minimum. This is the largest cost component. |
| **Bedrock (Embeddings)** | Charged per input token during ingestion and queries. Titan Embed v2 ≈ $0.02 per 1M input tokens. |
| **Bedrock (LLM)** | Charged per input/output token. Claude 3 Haiku ≈ $0.25/$1.25 per 1M input/output tokens. |
| **Lambda** | Free tier covers 1M requests/month. Minimal cost beyond that. |
| **API Gateway** | $3.50 per million API calls + data transfer. |
| **S3** | Standard storage pricing. Negligible for document storage. |

> **Tip:** For development/testing, consider destroying the stack when not in use to avoid OpenSearch Serverless costs.

---

## Files You Do NOT Need to Modify

These files are legacy/test scripts from the original local RAG setup and are **not used** by the Terraform deployment:

- `rag_backend.py` — Local LangChain backend (replaced by the Lambda function)
- `data_load_test.py` — PDF loader test
- `data_split_test.py` — Text splitter test
