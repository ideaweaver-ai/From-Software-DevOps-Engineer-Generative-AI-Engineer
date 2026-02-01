def build_devops_pr_review_prompt(pr_files):
    diffs = ""

    for f in pr_files:
        filename = f.get("filename", "unknown-file")
        patch = f.get("patch")

        diffs += f"### File: {filename}\n\n"

        if patch:
            safe_patch = patch.replace("```", "'''")
            diffs += "```diff\n" + safe_patch + "\n```\n"
        else:
            diffs += "No patch available (likely a binary or large file).\n"

        diffs += "\n---\n\n"

    user_message = f"""
You are a senior DevOps engineer with deep experience in CI/CD, IaC, Kubernetes, cloud, and security.

Please review the following code changes in these files:

{diffs}

Your mission:
- Review the proposed changes file by file.
- Focus on DevOps best practices, reliability, security, and scalability.
- Identify potential issues in CI/CD, IaC, containerization, or cloud configs.
- Suggest concrete improvements where applicable.
- Ignore files without meaningful patches.
"""
    return user_message.strip()


# n8n Python (Native): read input items
# Each incoming item looks like: {"json": {...}}
files = []
for it in _items:
    j = it.get("json", {})
    # Your HTTP node might return either the file object directly,
    # or wrap it under some key. This handles both.
    if isinstance(j, dict) and "filename" in j:
        files.append(j)
    elif isinstance(j, dict) and "body" in j and isinstance(j["body"], list):
        files.extend(j["body"])

prompt = build_devops_pr_review_prompt(files)

# n8n expects a list of output items
return [{"json": {"prompt": prompt}}]
