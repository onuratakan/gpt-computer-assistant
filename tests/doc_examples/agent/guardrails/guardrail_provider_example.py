from upsonic import Agent, Task
from upsonic.guardrails import AllowlistProvider
from upsonic.tools import tool


@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"Results for: {query}"


@tool
def delete_record(record_id: str) -> str:
    """Delete a record."""
    return f"Deleted: {record_id}"


agent = Agent(
    model="openai/gpt-4o",
    tools=[search_docs, delete_record],
    guardrail_provider=AllowlistProvider(allowed_tools=["search_docs"]),
)

task = Task("Search the docs for onboarding details")
result = agent.do(task)
print(result)
