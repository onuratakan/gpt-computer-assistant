"""GuardrailProvider patterns for common Upsonic use cases.

These examples intentionally use the built-in allowlist provider so they do not
require network calls or an external authorization service. Teams can replace
the provider with an internal policy service or an OAP/APort provider without
changing the Agent API.
"""

from upsonic.guardrails import AllowlistProvider


coding_workspace_guardrail = AllowlistProvider(
    allowed_tools=[
        "read_file",
        "search_files",
        "grep_files",
        "edit_file",
    ],
    denied_tools=[
        "delete_file",
        "run_command",
        "deploy_service",
    ],
)


merchant_ops_guardrail = AllowlistProvider(
    allowed_tools=[
        "classify_email",
        "extract_receipt",
        "analyze_merchant_document",
        "check_merchant_status",
    ],
    denied_tools=[
        "send_payout",
        "refund_payment",
        "export_customer_data",
    ],
)


contract_analysis_guardrail = AllowlistProvider(
    allowed_tools=[
        "ocr_document",
        "extract_contract_terms",
        "search_knowledge_base",
    ],
    denied_tools=[
        "send_email",
        "share_document",
        "export_customer_data",
    ],
)


research_agent_guardrail = AllowlistProvider(
    allowed_tools=[
        "web_search",
        "fetch_url",
        "summarize_page",
    ],
    denied_tools=[
        "send_message",
        "purchase_item",
        "change_account_settings",
    ],
)
