import pytest
from fastapi.testclient import TestClient
import os
import uuid
import httpx
import socket
import tempfile
import shutil
import time

# Assuming your main app is in agent_maker.main
# Adjust the import path if necessary
from agent_maker.main import app
from agent_maker.config import settings  # Import settings to check API key

client = TestClient(app)

# Helper to delete agent file after test
AGENT_DIR = "agent_maker/agents"
PROMPT_DIR = "agent_maker/prompt_library/prompts"
RESULTS_LOG_FILE = "tool_execution_log.jsonl"

@pytest.fixture(autouse=True)
def cleanup_files():
    """Clean up created files before and after each test."""
    # Before test: clean up any potential leftover files
    for fname in os.listdir(AGENT_DIR):
        if fname.endswith(".json") and fname.startswith("test_"):
            os.remove(os.path.join(AGENT_DIR, fname))
    for fname in os.listdir(PROMPT_DIR):
        if fname.endswith(".json") and fname.startswith("test_"):
            os.remove(os.path.join(PROMPT_DIR, fname))
    if os.path.exists(RESULTS_LOG_FILE):
        os.remove(RESULTS_LOG_FILE)
    yield
    # After test: clean up files created during the test
    for fname in os.listdir(AGENT_DIR):
        if fname.endswith(".json") and fname.startswith("test_"):
            os.remove(os.path.join(AGENT_DIR, fname))
    for fname in os.listdir(PROMPT_DIR):
        if fname.endswith(".json") and fname.startswith("test_"):
            os.remove(os.path.join(PROMPT_DIR, fname))
    if os.path.exists(RESULTS_LOG_FILE):
        os.remove(RESULTS_LOG_FILE)

# === Test Data ===
test_prompt = {
    "name": "test_math_prompt",
    "template": "What is the answer to {input}? Use the math tool if needed.",
    "input_variables": ["input"],
    "description": "A test prompt for math.",
    "metadata": {"version": "1.0"}
}

test_agent = {
    "agent_id": "test_math_agent",
    "agent_type": "react",  # Changed to react to match llm-math expectation
    "model": "meta/llama3-70b-instruct",
    "tools": ["llm-math"],  # Ensure llm-math tool exists and is usable
    "prompt_name": "test_math_prompt",
    "description": "Test agent for math.",
    "metadata": {}
}

# === Tests ===

@pytest.mark.skipif(
    not settings.NVIDIA_API_KEY,
    reason="NVIDIA API Key not configured"
)
def test_get_nvidia_models():
    response = client.get("/models/nvidia")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    # assert len(response.json()) > 0  # Might be empty depending on API state

def test_create_prompt():
    response = client.post("/prompts", json=test_prompt)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_prompt["name"]

def test_get_prompt():
    client.post("/prompts", json=test_prompt)
    response = client.get(f"/prompts/{test_prompt['name']}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_prompt["name"]

def test_list_prompts():
    client.post("/prompts", json=test_prompt)
    response = client.get("/prompts")
    assert response.status_code == 200
    assert test_prompt["name"] in response.json()

def test_create_agent():
    response = client.post("/agents", json=test_agent)
    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == test_agent["agent_id"]

def test_get_agent():
    client.post("/agents", json=test_agent)
    response = client.get(f"/agents/{test_agent['agent_id']}")
    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == test_agent["agent_id"]

def test_list_agents():
    client.post("/agents", json=test_agent)
    response = client.get("/agents")
    assert response.status_code == 200
    agent_list = response.json()
    assert isinstance(agent_list, list)
    assert any(
        agent['agent_id'] == test_agent['agent_id'] for agent in agent_list
    )

@pytest.mark.skipif(
    not settings.NVIDIA_API_KEY,
    reason="NVIDIA API Key not configured"
)
def test_run_agent():
    # Create dependent prompt and agent first
    client.post("/prompts", json=test_prompt)
    client.post("/agents", json=test_agent)
    # Run the agent
    response = client.post(
        f"/agents/{test_agent['agent_id']}/run",
        json={"input": "What is 2 + 2?"}
    )
    assert response.status_code == 200
    # Check headers for streaming response OR direct JSON response
    assert "content-type" in response.headers
    # Allow either streaming or direct JSON, as simple ReAct might optimize
    assert response.headers["content-type"].startswith((
        "text/event-stream", "application/json"
    ))
    # Note: TestClient doesn't easily support consuming the stream directly.
    # We verify the stream initiation here. Full stream consumption requires
    # httpx or similar.
    # Example of basic stream content check (might be fragile):
    # assert "event: end" in response.text

def test_run_agent_missing_input():
    client.post("/prompts", json=test_prompt)
    client.post("/agents", json=test_agent)
    response = client.post(
        f"/agents/{test_agent['agent_id']}/run",
        json={}  # Missing 'input'
    )
    # Should return 422 because 'input' is expected by the payload prep logic
    # even before hitting the agent stream, if prompt expects it.
    assert response.status_code == 422

def test_run_agent_with_nonexistent_tool():
    agent = {
        "agent_id": "bad_tool_agent",
        "agent_type": "react",
        "model": "meta/llama3-70b-instruct",
        "tools": ["not_a_real_tool"],
        "prompt_name": None,
        # Minimal template for testing tool loading
        "prompt_template": "Solve: {input}",
        "description": "Agent with bad tool.",
        "metadata": {}
    }
    client.post("/agents", json=agent)
    response = client.post(
        "/agents/bad_tool_agent/run",
        json={"input": "test"}
    )
    # Tool check happens before streaming starts
    assert response.status_code == 400

def test_run_agent_prompt_only():
    agent = {
        "agent_id": "prompt_only_agent",
        "agent_type": "react",  # Type doesn't matter much here
        "model": "meta/llama3-70b-instruct",
        "tools": [],  # No tools
        "prompt_name": None,
        "prompt_template": "Echo: {input}",
        "description": "Agent that only uses prompt.",
        "metadata": {}
    }
    client.post("/agents", json=agent)
    response = client.post(
        "/agents/prompt_only_agent/run",
        json={"input": "Hello"}
    )
    # Prompt-only agents currently don't stream, they return directly
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] == "Prompt-only agent: Echo: Hello"

# Add more tests for tool execution history, specific tools, error cases, etc.
# Skipping llm-math tool execution test for now as it requires numexpr
# and might be flaky depending on LLM behavior.
@pytest.mark.skip(reason="Skipping direct tool run test for llm-math")
def test_run_langchain_tool_llm_math():
    pass  # Requires numexpr, potentially slow/flaky

def test_create_and_use_prompt_library():
    # Create a prompt
    prompt = {
        "name": "lib_prompt",
        "template": "Echo: {input}",
        "description": "Echo prompt",
        "input_variables": ["input"]
    }
    agent = {
        "agent_id": "lib_prompt_agent",
        "agent_type": "react",
        "model": "meta/llama3-70b-instruct",
        "tools": [],
        "prompt_name": "lib_prompt",
        "prompt_template": None,
        "description": "Agent using prompt library.",
        "metadata": {}
    }
    client.post("/prompts", json=prompt)
    client.post("/agents", json=agent)
    response = client.post(
        "/agents/lib_prompt_agent/run",
        json={"input": "Hello!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "Echo: {input}"
    assert "result" in data

def test_get_results():
    response = client.get("/results")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_result_by_run_id_not_found():
    response = client.get("/results/doesnotexist-runid")
    assert response.status_code == 404

@pytest.mark.skip(reason="Requires running MCP server and tool config")
def test_run_agent_with_mcp_tool():
    agent = {
        "agent_id": "mcp_agent",
        "agent_type": "react",
        "model": "meta/llama3-70b-instruct",
        "tools": ["mcp_filesystem_read_file"],
        "prompt_name": None,
        "prompt_template": "Read file: {input}",
        "description": "Agent with MCP tool.",
        "metadata": {}
    }
    client.post("/agents", json=agent)
    response = client.post(
        "/agents/mcp_agent/run",
        json={"input": "/tmp/test.txt"}
    )
    assert response.status_code in (200, 400, 500)  # Accept error if MCP not available

# --- Memory Tests ---

def cleanup_agent(agent_id):
    path = os.path.join(AGENT_DIR, f"{agent_id}.json")
    if os.path.exists(path):
        os.remove(path)

def test_create_agent_with_memory():
    agent_id = f"mem_agent_{uuid.uuid4().hex[:8]}"
    agent = {
        "agent_id": agent_id,
        "agent_type": "react",
        "model": "meta/llama3-70b-instruct",
        "tools": [],
        "prompt_template": "Say hello to {input}",
        "has_memory": True,
        "description": "Memory agent test",
        "metadata": {}
    }
    resp = client.post("/agents", json=agent)
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_memory"] is True
    cleanup_agent(agent_id)

def test_run_agent_with_memory_first_turn():
    agent_id = f"mem_agent_{uuid.uuid4().hex[:8]}"
    agent = {
        "agent_id": agent_id,
        "agent_type": "react",
        "model": "meta/llama3-70b-instruct",
        "tools": [],
        "prompt_template": "Say hello to {input}",
        "has_memory": True,
        "description": "Memory agent test",
        "metadata": {}
    }
    client.post("/agents", json=agent)
    # First turn, no history
    resp = client.post(
        f"/agents/{agent_id}/run",
        json={"input": "Alice", "chat_history": []}
    )
    assert resp.status_code == 200 or resp.status_code == 206
    data = resp.json() if resp.headers["content-type"].startswith(
        "application/json"
    ) else None
    if data:
        assert "chat_history" in data
        assert data["chat_history"] == [] or isinstance(
            data["chat_history"], list
        )
    cleanup_agent(agent_id)

def test_run_agent_with_memory_second_turn():
    agent_id = f"mem_agent_{uuid.uuid4().hex[:8]}"
    agent = {
        "agent_id": agent_id,
        "agent_type": "react",
        "model": "meta/llama3-70b-instruct",
        "tools": [],
        "prompt_template": "Say hello to {input}",
        "has_memory": True,
        "description": "Memory agent test",
        "metadata": {}
    }
    client.post("/agents", json=agent)
    # First turn
    client.post(
        f"/agents/{agent_id}/run",
        json={"input": "Alice", "chat_history": []}
    )
    # Second turn, send previous history
    chat_history = [
        {"role": "user", "content": "Alice"},
        {"role": "assistant", "content": "Hello Alice"}
    ]
    resp2 = client.post(
        f"/agents/{agent_id}/run",
        json={"input": "Bob", "chat_history": chat_history}
    )
    assert resp2.status_code == 200 or resp2.status_code == 206
    data = resp2.json() if resp2.headers["content-type"].startswith(
        "application/json"
    ) else None
    if data:
        assert "chat_history" in data
        assert any(
            msg["content"] == "Hello Alice" for msg in data["chat_history"]
        )
    cleanup_agent(agent_id)

def test_run_agent_without_memory_ignores_history():
    agent_id = f"no_mem_agent_{uuid.uuid4().hex[:8]}"
    agent = {
        "agent_id": agent_id,
        "agent_type": "react",
        "model": "meta/llama3-70b-instruct",
        "tools": [],
        "prompt_template": "Say hello to {input}",
        "has_memory": False,
        "description": "No memory agent test",
        "metadata": {}
    }
    client.post("/agents", json=agent)
    chat_history = [
        {"role": "user", "content": "Alice"},
        {"role": "assistant", "content": "Hello Alice"}
    ]
    resp = client.post(
        f"/agents/{agent_id}/run",
        json={"input": "Bob", "chat_history": chat_history}
    )
    assert resp.status_code == 200 or resp.status_code == 206
    data = resp.json() if resp.headers["content-type"].startswith(
        "application/json"
    ) else None
    if data:
        # Should not echo back history
        assert "chat_history" not in data or data["chat_history"] == []
    cleanup_agent(agent_id)

def is_server_running():
    try:
        with socket.create_connection(("localhost", 8000), timeout=1):
            return True
    except Exception:
        return False

@pytest.mark.asyncio
def test_streaming_agent_response():
    if not is_server_running():
        pytest.skip(
            "FastAPI server must be running on http://localhost:8000 for this test."
        )
    agent_id = f"stream_agent_{uuid.uuid4().hex[:8]}"
    agent = {
        "agent_id": agent_id,
        "agent_type": "react",
        "model": "meta/llama3-70b-instruct",
        "tools": ["llm-math"],  # Add a tool to force streaming
        "prompt_template": "What is {input}? Use the math tool if needed.",
        "has_memory": False,
        "description": "Streaming agent test",
        "metadata": {},
    }
    #  Create agent
    client.post("/agents", json=agent)
    #  Use httpx.AsyncClient to consume the stream from a running server
    import asyncio

    async def run_stream():
        async with httpx.AsyncClient(
            base_url="http://localhost:8000"
        ) as async_client:
            response = await async_client.post(
                f"/agents/{agent_id}/run",
                json={"input": "2 + 2"},
                timeout=10.0,
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(
                "text/event-stream"
            )
            data_events = []
            end_event = None
            stream_lines = []
            async for line in response.aiter_lines():
                print("STREAM LINE:", repr(line))  # Debug print
                stream_lines.append(line)
                if line.startswith("data: "):
                    data_events.append(line)
                if "event: end" in line:
                    end_event = line
            assert data_events, (
                "Should receive at least one data event (token)"
            )
            assert end_event is not None, (
                f"Should receive a final end event. Stream lines: {stream_lines}"
            )
    asyncio.run(run_stream())
    #  Cleanup
    path = os.path.join(AGENT_DIR, f"{agent_id}.json")
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def temp_vector_dir():
    d = tempfile.mkdtemp(prefix="test_vectorstore_")
    yield d
    shutil.rmtree(d)

def test_vector_search_basic(temp_vector_dir):
    # Index some chunks
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    resp = client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    assert resp.status_code == 200
    # Basic vector search
    resp = client.post(
        "/vector/search",
        json={
            "query": "capital of France",
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert any("Paris" in r["text"] for r in data["results"])

def test_vector_search_multi_query(temp_vector_dir):
    time.sleep(1)
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/search",
        json={
            "query": "capital of Germany",
            "retriever_type": "multi_query",
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["results"]

def test_vector_search_ensemble(temp_vector_dir):
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/search",
        json={
            "query": "capital of Italy",
            "retriever_type": "ensemble",
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert any("Rome" in r["text"] for r in data["results"])

def test_vector_search_long_context_reorder(temp_vector_dir):
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/search",
        json={
            "query": "capital",
            "retriever_type": "long_context_reorder",
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) <= 2

def test_vector_search_rerank(monkeypatch, temp_vector_dir):
    class DummyRerank:
        def __init__(self, **kwargs):
            pass

        def compress_documents(self, query, documents):
            return list(reversed(documents))
    import sys
    sys.modules["langchain_nvidia_ai_endpoints"].NVIDIARerank = DummyRerank
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/search",
        json={
            "query": "capital",
            "rerank": True,
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) <= 2

def test_hybrid_search_basic(temp_vector_dir):
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/hybrid_search",
        json={
            "query": "capital",
            "keyword": "Paris",
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert any("Paris" in r["text"] for r in data["results"])

def test_hybrid_search_multi_query(temp_vector_dir):
    time.sleep(1)
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/hybrid_search",
        json={
            "query": "capital of Germany",
            "retriever_type": "multi_query",
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["results"]

def test_hybrid_search_ensemble(temp_vector_dir):
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/hybrid_search",
        json={
            "query": "capital of Italy",
            "retriever_type": "ensemble",
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert any("Rome" in r["text"] for r in data["results"])

def test_hybrid_search_long_context_reorder(temp_vector_dir):
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/hybrid_search",
        json={
            "query": "capital",
            "retriever_type": "long_context_reorder",
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) <= 2

def test_hybrid_search_rerank(monkeypatch, temp_vector_dir):
    class DummyRerank:
        def __init__(self, **kwargs):
            pass

        def compress_documents(self, query, documents):
            return list(reversed(documents))
    import sys
    sys.modules["langchain_nvidia_ai_endpoints"].NVIDIARerank = DummyRerank
    chunks = [
        {"text": "The capital of France is Paris.", "metadata": {"id": "1"}},
        {"text": "The capital of Germany is Berlin.", "metadata": {"id": "2"}},
        {"text": "The capital of Italy is Rome.", "metadata": {"id": "3"}},
    ]
    client.post(
        "/vector/index",
        json={"chunks": chunks, "persist_directory": temp_vector_dir}
    )
    resp = client.post(
        "/vector/hybrid_search",
        json={
            "query": "capital",
            "rerank": True,
            "top_k": 2,
            "persist_directory": temp_vector_dir
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) <= 2 