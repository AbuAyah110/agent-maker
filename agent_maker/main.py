from fastapi import (
    FastAPI, HTTPException, status, Request, Query, Body, UploadFile, File
)
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Union
import importlib
import asyncio
import traceback
import uuid
import json
from datetime import datetime
import os
from pydantic import BaseModel, Field
import httpx
from langchain.tools import Tool
from fastapi.middleware.cors import CORSMiddleware

from agent_maker.model_manager.nvidia_nims import (
    get_available_nvidia_models
)
from agent_maker.config import settings
# Import prompt library functions, Pydantic model, and errors
from agent_maker.prompt_library.manager import (
    list_prompts,
    load_prompt,
    save_prompt,
    PromptData,
    PromptNotFoundError,
    PromptSaveError
)
# Import tool library functions and models
from agent_maker.tool_library.manager import (
    list_langchain_tools,
    get_langchain_tool_details,
    list_custom_tools,
    load_custom_tool,
    # save_custom_tool, # Not creating POST endpoint for custom tools yet
    LangChainToolData,
    CustomToolData,  # For response model, though load_custom_tool placeholder
    ToolNotFoundError
)

# Add for MCP tool execution
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
except ImportError:
    MultiServerMCPClient = None
    load_mcp_tools = None

from langchain_nvidia_ai_endpoints import (
    ChatNVIDIA, NVIDIAEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_tool_calling_agent
from langchain.chains.llm_math.base import LLMMathChain
from agent_maker.model_manager.nvidia_embeddings import NvidiaEmbeddingService
from agent_maker.model_manager.document_loader import load_document
from agent_maker.model_manager.chunking import Chunker, ChunkingStrategy
from agent_maker.model_manager.vector_store import VectorStoreManager
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

app = FastAPI(
    title="Agent Maker API",
    description=(
        "API for creating and managing AI agents."
    ),
    version="0.1.0",
)

# Add CORS middleware
origins = [
    "http://localhost",      # Allow requests from base localhost
    "http://localhost:3000", # Default Next.js dev port
    # Add any other origins if needed (e.g., your deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

RESULTS_LOG_FILE = "tool_execution_log.jsonl"

def log_tool_execution(tool_type, tool_name, params, result=None, error=None):
    log_entry = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "tool_type": tool_type,
        "tool_name": tool_name,
        "params": params,
        "result": result,
        "error": format_error(error) if error else None,
    }
    print(f"[TOOL EXECUTION] {json.dumps(log_entry, indent=2)}")
    try:
        with open(RESULTS_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[LOGGING ERROR] Could not write to log file: {e}")
    return log_entry["run_id"]

def format_error(error):
    if error is None:
        return None
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }

@app.on_event("startup")
async def startup_event():
    # You can add any startup logic here, e.g., initial checks
    if not settings.NVIDIA_API_KEY:
        print(
            "WARNING: NVIDIA_API_KEY is not configured in .env. "
            "Model listing will likely fail."
        )
    else:
        print("NVIDIA_API_KEY found. Ready to fetch models.")


# === Model Endpoints ===

@app.get(
    "/models/nvidia",
    response_model=List[str],
    summary="List available NVIDIA models",
    tags=["Models"]
)
async def get_nvidia_models():
    """Returns a list of available NVIDIA NIMs."""
    try:
        # Ensure NVIDIA_API_KEY is loaded via settings
        if not settings.NVIDIA_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="NVIDIA_API_KEY not configured."
            )
        models = get_available_nvidia_models()
        if not models:
            # This case might occur if the API returns an empty list
            return []
        return models
    except ValueError as ve:
        # Specific error from get_available_nvidia_models if key is missing
        # and running in a non-interactive environment.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        # Catch-all for other errors from the NVIDIA SDK or network issues
        print(f"Error fetching NVIDIA models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve models from NVIDIA: {str(e)}"
        )


# === Prompt Library Endpoints ===

@app.get(
    "/prompts",
    response_model=List[str],
    summary="List available prompt templates",
    tags=["Prompts"]
)
async def list_prompts_endpoint():
    """
    Retrieves a list of names for all available prompt templates
    (JSON and YAML).
    """
    try:
        return list_prompts()
    except Exception as e:
        print(f"Error listing prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list prompts."
        )


@app.get(
    "/prompts/{prompt_name}",
    response_model=Dict[str, Any],
    summary="Get a specific prompt template",
    tags=["Prompts"],
)
async def get_prompt_endpoint(prompt_name: str):
    """
    Retrieves the content of a specific prompt template by its name.
    The server will attempt to load from .json, .yaml, or .yml file types.
    """
    try:
        return load_prompt(prompt_name)
    except PromptNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        )
    except (ValueError, IOError) as e:  # Handle JSON/YAML decode or read errors
        print(f"Error loading prompt '{prompt_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Error processing prompt file '{prompt_name}'."
            ),
        )
    except Exception as e:
        print(f"Unexpected error loading prompt '{prompt_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error loading prompt.",
        )


@app.post(
    "/prompts",
    response_model=PromptData,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update a prompt template",
    tags=["Prompts"]
)
async def create_prompt_endpoint(prompt: PromptData):
    """
    Creates a new prompt template or updates an existing one (based on name).
    The prompt will be saved as a JSON file.

    Request body must contain: `name`, `template`.
    Optional fields: `description`, `input_variables`, `metadata`.
    """
    try:
        saved_prompt_data = save_prompt(prompt)
        return saved_prompt_data
    except PromptSaveError as e:
        print(f"Error saving prompt '{prompt.name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save prompt '{prompt.name}': {str(e)}"
        )
    except Exception as e:
        # Catch any other unexpected errors during the save process
        print(
            f"Unexpected error creating/updating prompt '{prompt.name}': {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error while saving prompt.",
        )


# === Tool Library Endpoints ===

@app.get(
    "/tools/langchain",
    response_model=List[LangChainToolData],
    summary="List available LangChain tools (curated)",
    tags=["Tools"]
)
async def list_langchain_tools_endpoint():
    """Retrieves a curated list of available LangChain tools with details."""
    try:
        tools = list_langchain_tools()
        return tools
    except Exception as e:
        print(f"Error listing LangChain tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Failed to list LangChain tools."
            ),
        )


@app.get(
    "/tools/langchain/{tool_name}",
    response_model=LangChainToolData,
    summary="Get details for a specific LangChain tool",
    tags=["Tools"]
)
async def get_langchain_tool_endpoint(tool_name: str):
    """Retrieves details for a specific LangChain tool from curated list."""
    try:
        tool_details = get_langchain_tool_details(tool_name)
        if not tool_details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"LangChain tool '{tool_name}' not found in curated list."
                ),
            )
        return tool_details
    except Exception as e:  # Catch any other unexpected error from the manager
        print(f"Error getting LangChain tool '{tool_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve LangChain tool '{tool_name}'.",
        )


@app.get(
    "/tools/custom",
    response_model=List[CustomToolData],
    summary="List available custom tools",
    tags=["Tools"]
)
async def list_custom_tools_endpoint():
    """Retrieves a list of available custom tools with their metadata."""
    try:
        # Load all custom tool metadata (not just names)
        custom_tools = []
        for tool in list_custom_tools():
            # list_custom_tools should return dicts or CustomToolData
            if isinstance(tool, dict):
                custom_tools.append(CustomToolData(**tool))
            elif isinstance(tool, CustomToolData):
                custom_tools.append(tool)
        return custom_tools
    except Exception as e:
        print(f"Error listing custom tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list custom tools."
        )


@app.get(
    "/tools/custom/{tool_name}",
    response_model=CustomToolData,
    summary="Get details for a specific custom tool",
    tags=["Tools"]
)
async def get_custom_tool_endpoint(tool_name: str):
    """Retrieves details for a specific custom tool."""
    try:
        tool_data = load_custom_tool(tool_name)
        if not tool_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Custom tool '{tool_name}' not found or not implemented."
            )
        # Assuming load_custom_tool will eventually return a dict or Pydantic model
        # For now, if it returns something, wrap it for consistency if it's a dict.
        if isinstance(tool_data, dict):
            return CustomToolData(**tool_data)
        elif isinstance(tool_data, CustomToolData):
            return tool_data
        else:
            # If it's some other type, how to handle? For now, raise error.
            # This part needs to be robust based on actual load_custom_tool behavior.
            print(
                f"Warning: Custom tool '{tool_name}' loaded but returned "
                f"unexpected type: {type(tool_data)}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Custom tool '{tool_name}' loaded but in unexpected format."
            )

    except ToolNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        )
    except Exception as e:
        print(f"Error getting custom tool '{tool_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve custom tool '{tool_name}'.",
        )


@app.get(
    "/tools",
    response_model=Dict[str, List[Union[LangChainToolData, CustomToolData]]],
    summary="List all available tools (LangChain and custom)",
    tags=["Tools"]
)
async def list_all_tools_endpoint():
    """Returns all available tools, grouped by type ('langchain', 'custom')."""
    try:
        langchain_tools = list_langchain_tools()
        custom_tools = []
        for tool in list_custom_tools():
            if isinstance(tool, dict):
                custom_tools.append(CustomToolData(**tool))
            elif isinstance(tool, CustomToolData):
                custom_tools.append(tool)
        return {"langchain": langchain_tools, "custom": custom_tools}
    except Exception as e:
        print(f"Error listing all tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list all tools."
        )


@app.post(
    "/tools/langchain/{tool_name}/run",
    summary="Execute a LangChain tool asynchronously",
    tags=["Tools"]
)
async def run_langchain_tool_endpoint(tool_name: str, request: Request):
    """Executes a LangChain tool with provided parameters."""
    try:
        tool = get_langchain_tool_details(tool_name)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LangChain tool '{tool_name}' not found."
            )
        params = await request.json()
        # Dynamically import the tool class/function
        module_path = tool.metadata.langchain_module
        class_name = tool.usage_class_or_function
        if not module_path or not class_name:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tool metadata missing module or class/function name."
            )
        mod = importlib.import_module(module_path)
        ToolClass = getattr(mod, class_name)
        # Instantiate tool (assume all params are passed in params dict)
        # Some tools may require special handling; this is a generic approach
        tool_instance = ToolClass(**params)
        # Most LangChain tools have a .run() or __call__ method
        if hasattr(tool_instance, "run"):
            if asyncio.iscoroutinefunction(tool_instance.run):
                result = await tool_instance.run(**params)
            else:
                result = tool_instance.run(**params)
        elif callable(tool_instance):
            if asyncio.iscoroutinefunction(tool_instance):
                result = await tool_instance(**params)
            else:
                result = tool_instance(**params)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tool is not executable."
            )
        run_id = log_tool_execution("langchain", tool_name, params, result)
        return {"result": result, "run_id": run_id}
    except Exception as e:
        print(f"Error running LangChain tool '{tool_name}': {e}")
        run_id = log_tool_execution("langchain", tool_name, params, error=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute LangChain tool '{tool_name}': {str(e)}",
            headers={"X-Run-ID": run_id}
        )

@app.post(
    "/tools/custom/{tool_name}/run",
    summary="Execute an MCP tool asynchronously via langchain-mcp-adapters",
    tags=["Tools"]
)
async def run_mcp_tool_endpoint(tool_name: str, request: Request):
    """Executes an MCP tool with params using langchain-mcp-adapters."""
    if MultiServerMCPClient is None or load_mcp_tools is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="langchain-mcp-adapters is not installed."
        )
    try:
        params = await request.json()
        # For now, assume a single local MCP server (can be extended)
        # You may want to pass server info in params in the future
        mcp_server_config = {
            "filesystem": {
                "url": (
                    "http://localhost:8001/sse"  # Example, adjust as needed
                ),
                "transport": "sse",
            }
        }
        # Always define mcp_tool_objs before use
        mcp_tool_objs = []
        mcp_tools = []  # Fix: define mcp_tools as empty list
        if mcp_tools and MultiServerMCPClient and load_mcp_tools:
            async with MultiServerMCPClient(mcp_server_config) as client:
                all_mcp_tools = client.get_tools()
                for t in all_mcp_tools:
                    if hasattr(t, "name") and t.name in mcp_tools:
                        mcp_tool_objs.append(t)
        # Only use mcp_tool_objs if not empty
        if mcp_tool_objs:
            if asyncio.iscoroutinefunction(mcp_tool_objs[0]):
                result = await mcp_tool_objs[0](**params)
            else:
                result = mcp_tool_objs[0](**params)
            run_id = log_tool_execution("mcp", tool_name, params, result)
            return {"result": result, "run_id": run_id}
    except (httpx.ConnectError, Exception) as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to connect to MCP server or tool not found: {str(e)}"
            ),
        )

@app.get(
    "/results",
    summary="Get tool execution history",
    tags=["Results"]
)
async def get_results(tool_type: str = None, tool_name: str = None, error_only: bool = False):
    """
    Returns the tool execution history. Supports optional filtering by tool_type, tool_name, and error presence.
    """
    results = []
    try:
        with open(RESULTS_LOG_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                if tool_type and entry.get("tool_type") != tool_type:
                    continue
                if tool_name and entry.get("tool_name") != tool_name:
                    continue
                if error_only and not entry.get("error"):
                    continue
                results.append(entry)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[RESULTS ERROR] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read results: {str(e)}"
        )
    return results

@app.get(
    "/results/{run_id}",
    summary="Get a single tool execution result by run_id",
    tags=["Results"]
)
async def get_result_by_id(run_id: str):
    """
    Returns a single tool execution result by its run_id (UUID).
    """
    try:
        with open(RESULTS_LOG_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("run_id") == run_id:
                    return entry
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result with run_id {run_id} not found."
        )
    except Exception as e:
        print(f"[RESULTS ERROR] {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read result: {str(e)}"
        )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Result with run_id {run_id} not found."
    )

# To run this app (from the project root,
# e.g., /Users/mraza/Documents/src/agent_maker/):
# uvicorn agent_maker.main:app --reload


if __name__ == "__main__":
    # This section is for debugging and direct execution,
    # but uvicorn is the standard way to run FastAPI apps.
    import uvicorn
    print("Running FastAPI app with uvicorn for local testing.")
    print("Create a .env file in the project root with your NVIDIA_API_KEY.")
    print("Access the API docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

AGENTS_DIR = os.path.join(os.path.dirname(__file__), "agents")

class AgentDefinition(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(
        ..., description="Type of agent: 'react' or 'tool-calling'"
    )
    model: str = Field(
        ..., description="Model name or config (e.g., 'gpt-4o')"
    )
    tools: list = Field(
        ..., description="List of tool names (LangChain and/or MCP)"
    )
    prompt_name: str | None = Field(
        None, description="Prompt template name from prompt library"
    )
    prompt_template: str | None = Field(
        None,
        description="Custom prompt template (overrides prompt_name if set)",
    )
    has_memory: bool = Field(
        False, description="Whether the agent should use conversation memory"
    )
    description: str | None = None
    metadata: dict = Field(default_factory=dict)

# Ensure agents directory exists
os.makedirs(AGENTS_DIR, exist_ok=True)

def agent_file_path(agent_id):
    return os.path.join(AGENTS_DIR, f"{agent_id}.json")

@app.post(
    "/agents",
    response_model=AgentDefinition,
    summary="Create or update an agent definition",
    tags=["Agents"]
)
async def create_agent_endpoint(agent: AgentDefinition):
    """Create or update an agent definition (stored as JSON)."""
    try:
        with open(agent_file_path(agent.agent_id), "w") as f:
            f.write(agent.model_dump_json(indent=2))
        return agent
    except Exception as e:
        print(f"Error saving agent '{agent.agent_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save agent '{agent.agent_id}': {str(e)}"
        )

@app.get(
    "/agents",
    response_model=list[AgentDefinition],
    summary="List all agent definitions",
    tags=["Agents"]
)
async def list_agents_endpoint():
    """List all agent definitions."""
    agents = []
    try:
        for fname in os.listdir(AGENTS_DIR):
            if fname.endswith(".json"):
                with open(os.path.join(AGENTS_DIR, fname), "r") as f:
                    agents.append(AgentDefinition.model_validate_json(f.read()))
        return agents
    except Exception as e:
        print(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list agents."
        )

@app.get(
    "/agents/{agent_id}",
    response_model=AgentDefinition,
    summary="Get a specific agent definition",
    tags=["Agents"]
)
async def get_agent_endpoint(agent_id: str):
    """Get a specific agent definition by agent_id."""
    try:
        with open(agent_file_path(agent_id), "r") as f:
            return AgentDefinition.model_validate_json(f.read())
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Agent '{agent_id}' not found."
            )
        )
    except Exception as e:
        print(f"Error loading agent '{agent_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Failed to load agent '{agent_id}': {str(e)}"
            )
        )

@app.post(
    "/agents/{agent_id}/run",
    summary="Run an agent workflow with user input",
    tags=["Agents"]
)
async def run_agent_endpoint(agent_id: str, request: Request) -> StreamingResponse:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    """
    Run an agent workflow (ReAct or tool-calling) with user input.
    This will support both LangChain and MCP tools, prompt library integration, and memory.
    Returns a StreamingResponse with Server-Sent Events.
    """
    try:
        with open(agent_file_path(agent_id), "r") as f:
            agent_def = AgentDefinition.model_validate_json(f.read())
        user_input_data = await request.json()

        # Extract chat_history from user_input_data for memory-enabled agents
        # Default to an empty list if not provided
        current_chat_history_messages = []
        if agent_def.has_memory:
            # We expect chat_history to be a list of dicts like {"role": "user"/"assistant", "content": "..."}
            # or already a list of BaseMessage objects if the client is sophisticated.
            # For ConversationBufferMemory, we'll need to convert dicts to BaseMessages.
            raw_history = user_input_data.get("chat_history", [])
            for msg_data in raw_history:
                if isinstance(msg_data, BaseMessage):
                    current_chat_history_messages.append(msg_data)
                elif isinstance(msg_data, dict):
                    role = msg_data.get("role")
                    content = msg_data.get("content")
                    if role == "user":
                        current_chat_history_messages.append(HumanMessage(content=content))
                    elif role == "assistant" or role == "ai":
                        current_chat_history_messages.append(AIMessage(content=content))
                    elif role == "system":
                        current_chat_history_messages.append(SystemMessage(content=content))
                    # Silently skip malformed history items for now
            user_input = user_input_data.get("input") # Main input query
        else:
            user_input = user_input_data # If no memory, user_input is the whole request body

        # 3. Instantiate model (move this before tool loading)
        llm = ChatNVIDIA(
            model=agent_def.model,
            api_key=settings.NVIDIA_API_KEY
        )
        # 1. Load prompt
        prompt = None
        if agent_def.prompt_template:
            prompt = agent_def.prompt_template
        elif agent_def.prompt_name:
            try:
                prompt_data = load_prompt(agent_def.prompt_name)
                prompt = prompt_data.get("template")
            except Exception as e:
                print(f"[AGENT] Warning: Could not load prompt '{agent_def.prompt_name}': {e}")
        # 2. Load tools
        langchain_tools = []
        mcp_tools = []
        mcp_server_config = {
            "filesystem": {
                "url": "http://localhost:8001/sse",
                "transport": "sse"
            }
        }
        # Load LangChain tools
        for tool_name in agent_def.tools:
            tool_data = get_langchain_tool_details(tool_name)
            if tool_data:
                try:
                    # Special handling for llm-math
                    if tool_name == "llm-math":
                        try:
                            # Check if numexpr is available by trying to create the chain
                            math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
                            langchain_tools.append(
                                Tool(
                                    name="llm-math",
                                    func=math_chain.run,
                                    description="Useful for when you need to answer quantitative questions."
                                )
                            )
                            print(f"[AGENT] Successfully loaded LangChain tool: {tool_name}")
                        except ImportError:
                            print(f"[AGENT] Warning: Could not load LangChain tool '{tool_name}': requires `numexpr` package.")
                        except Exception as math_e:
                             print(f"[AGENT] Warning: Error initializing LLMMathChain for tool '{tool_name}': {math_e}")
                    else:
                        # Generic tool loading
                        module_path = tool_data.metadata.langchain_module
                        class_name = tool_data.usage_class_or_function
                        mod = importlib.import_module(module_path)
                        ToolClass = getattr(mod, class_name)
                        # Basic instantiation - might need adjustment for tools requiring args
                        langchain_tools.append(ToolClass())
                        print(f"[AGENT] Successfully loaded LangChain tool: {tool_name}")

                except Exception as e:
                    print(f"[AGENT] Warning: Could not load/initialize LangChain tool '{tool_name}': {e}")
            else:
                # Try as MCP tool
                mcp_tools.append(tool_name)
        # If no tools and prompt is present, allow prompt-only agent
        if not langchain_tools and not mcp_tools and prompt:
            # Check for required input variable if prompt expects 'input'
            # For prompt-only with memory, ensure chat_history doesn't break .format()
            # Simplification: prompt-only agents typically don't use deep memory integration here.
            if prompt and '{input}' in prompt and (not user_input or (isinstance(user_input, dict) and 'input' not in user_input)):
                # If user_input became a string from the memory logic, check it directly
                if not isinstance(user_input, str):
                    raise HTTPException(
                        status_code=422,
                        detail="Missing required input variable: 'input' for prompt-only agent."
                    )
            
            formatted_prompt_input = user_input.get('input', '') if isinstance(user_input, dict) else user_input

            return {
                "agent_id": agent_id,
                "agent_type": agent_def.agent_type,
                "model": agent_def.model,
                "tools": agent_def.tools,
                "prompt": prompt,
                "user_input": user_input_data,
                "result": f"Prompt-only agent: {prompt.format(input=formatted_prompt_input)}",
                "chat_history": current_chat_history_messages # Return history even for prompt-only if memory was on
            }
        # If there are tools that are not found in either LangChain or MCP, return 400
        if mcp_tools and (MultiServerMCPClient is None or load_mcp_tools is None):
            raise HTTPException(
                status_code=400,
                detail=f"Custom MCP tools not found or MCP client not installed: {mcp_tools}"
            )
        # 4. Build prompt template (if any)
        prompt_obj = None
        if prompt and ChatPromptTemplate:
            prompt_obj = ChatPromptTemplate.from_template(prompt)
        # 5. Build agent
        # Always define mcp_tool_objs before use
        mcp_tool_objs = []
        try:
            if mcp_tools and MultiServerMCPClient and load_mcp_tools:
                async with MultiServerMCPClient(mcp_server_config) as client:
                    all_mcp_tools = client.get_tools()
                    for t in all_mcp_tools:
                        if hasattr(t, "name") and t.name in mcp_tools:
                            mcp_tool_objs.append(t)
        except (httpx.ConnectError, Exception) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to connect to MCP server or tool not found: {str(e)}"
            )
        tools = langchain_tools + mcp_tool_objs
        if not tools:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid tools found for this agent."
            )
        if agent_def.agent_type == "react":
            if not create_react_agent:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="langgraph is not installed."
                )
            
            react_prompt_to_use = None
            if prompt_obj:
                expected_vars = {'input', 'agent_scratchpad'}
                if expected_vars.issubset(prompt_obj.input_variables):
                    react_prompt_to_use = prompt_obj
                else:
                    print(
                        f"[AGENT] Warning: Custom prompt for ReAct agent '{agent_id}' "
                        f"does not contain required variables {expected_vars}. "
                        "Using default ReAct prompt."
                    )
            agent = create_react_agent(llm, tools, prompt=react_prompt_to_use)

        elif agent_def.agent_type == "tool-calling":
            if not create_tool_calling_agent:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="langgraph is not installed."
                )

            tool_calling_prompt_to_use = None
            if prompt_obj:
                # Tool-calling agents expect 'messages' and often 'agent_scratchpad' (implicitly via MessagesPlaceholder)
                # A simple check could be if 'messages' is an input variable.
                # More robustly, the prompt should be a ChatPromptTemplate with appropriate placeholders.
                if 'messages' in prompt_obj.input_variables:
                    tool_calling_prompt_to_use = prompt_obj
                else:
                    print(
                        f"[AGENT] Warning: Custom prompt for tool-calling agent '{agent_id}' "
                        f"does not seem to be a valid messages-based prompt. "
                        "Using default tool-calling prompt."
                    )
            agent = create_tool_calling_agent(llm, tools, prompt=tool_calling_prompt_to_use)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown agent_type: {agent_def.agent_type}"
            )
        # 6. Run agent with user input
        ainvoke_payload = None
        if prompt_obj and hasattr(prompt_obj, "input_variables"):
            input_vars = set(prompt_obj.input_variables)
            # Check for required input variables
            # The main query for agents with memory is now in `user_input` (string), not `user_input_data` (dict)
            # For agents without memory, user_input is user_input_data (dict)
            check_source = user_input_data if not agent_def.has_memory else {"input": user_input} 
            
            missing_vars = [var for var in input_vars if var not in check_source and var != 'chat_history' and var != 'agent_scratchpad']
            if missing_vars:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing required input variable(s): {missing_vars} in {check_source}"
                )
            
            # Determine ainvoke_payload based on prompt vars and available data
            if "input" in input_vars and "input" in check_source:
                ainvoke_payload = {"input": check_source["input"]}
            elif "messages" in input_vars and "messages" in check_source: # Typically for tool-calling without memory / explicit messages
                ainvoke_payload = {"messages": check_source["messages"]}
            # If memory is enabled, chat_history is handled separately and injected into final_payload

        if ainvoke_payload is None:
            if isinstance(user_input, str): # This implies memory is enabled and user_input is the query
                 ainvoke_payload = {"input": user_input}
            elif isinstance(user_input_data, dict) and "input" in user_input_data:
                ainvoke_payload = {"input": user_input_data["input"]}
            elif isinstance(user_input_data, dict) and "messages" in user_input_data:
                ainvoke_payload = {"messages": user_input_data["messages"]}
            else:
                ainvoke_payload = user_input_data # pass the whole dict

        final_payload = {}
        if agent_def.agent_type == "react":
            # Always provide both 'input' and 'messages' for compatibility
            if isinstance(user_input, str):
                final_payload["input"] = user_input
                final_payload["messages"] = [HumanMessage(content=user_input)]
            elif isinstance(ainvoke_payload, dict) and "input" in ainvoke_payload:
                final_payload["input"] = str(ainvoke_payload["input"])
                final_payload["messages"] = [HumanMessage(content=ainvoke_payload["input"])]
            elif isinstance(user_input_data, dict) and "input" in user_input_data:
                final_payload["input"] = str(user_input_data["input"])
                final_payload["messages"] = [HumanMessage(content=user_input_data["input"])]
            else:
                final_payload["input"] = str(user_input_data)
                final_payload["messages"] = [HumanMessage(content=str(user_input_data))]
                print(f"[AGENT] Warning: ReAct agent '{agent_id}' input coercion. Input: {user_input_data}")
            final_payload["chat_history"] = current_chat_history_messages # Use parsed BaseMessages

        elif agent_def.agent_type == "tool-calling":
            if agent_def.has_memory:
                # Construct messages: history + new human input
                constructed_messages = list(current_chat_history_messages) # Start with existing history
                if isinstance(user_input, str):
                    constructed_messages.append(HumanMessage(content=user_input))
                elif isinstance(user_input, dict) and "input" in user_input: # Should not happen if memory path taken
                    constructed_messages.append(HumanMessage(content=user_input["input"]))
                final_payload["messages"] = constructed_messages
            else: # No memory, direct payload construction as before
                if "messages" in user_input_data:
                    final_payload["messages"] = user_input_data["messages"]
                elif "input" in user_input_data:
                    final_payload["messages"] = [HumanMessage(content=user_input_data["input"])]
                else:
                    # Fallback: if ainvoke_payload is missing messages, create a default one
                    if isinstance(ainvoke_payload, dict) and "messages" in ainvoke_payload and ainvoke_payload["messages"]:
                        final_payload["messages"] = ainvoke_payload["messages"]
                    elif isinstance(ainvoke_payload, dict) and "input" in ainvoke_payload:
                        final_payload["messages"] = [HumanMessage(content=ainvoke_payload["input"])]
                    else:
                        # As a last resort, raise a clear error
                        raise HTTPException(
                            status_code=422,
                            detail="No valid 'input' or 'messages' found for tool-calling agent. At least one message is required."
                        )
        else: 
            final_payload = ainvoke_payload

        # --- Streaming Logic ---        
        async def stream_generator():
            accumulated_final_output = None
            updated_history_for_stream = list(current_chat_history_messages) # Start with initial history
            latest_human_input = None
            if agent_def.has_memory and isinstance(user_input, str):
                latest_human_input = HumanMessage(content=user_input)
                # Add user input to history *before* streaming AI response chunks
                if not updated_history_for_stream or updated_history_for_stream[-1] != latest_human_input:
                    updated_history_for_stream.append(latest_human_input)

            try:
                async for chunk in agent.astream(final_payload):
                    # LangGraph stream chunks are dictionaries, often nested.
                    # We need to parse them to find relevant info (LLM tokens, final output).
                    # This parsing logic might need refinement based on actual agent output structure.
                    
                    llm_token = None
                    # --- Example Parsing Logic (Needs Adjustment based on LangGraph version/agent) ---
                    # Heuristic: Look for common patterns in ReAct/Tool-calling agent streams
                    if isinstance(chunk, dict):
                        # Check for final answer patterns (often under 'agent' or specific node names)
                        agent_node = chunk.get("agent", chunk.get("__end__")) # Check common keys
                        if isinstance(agent_node, dict) and ("output" in agent_node or "messages" in agent_node):
                             if agent_def.agent_type == "react" and "output" in agent_node:
                                 accumulated_final_output = agent_node["output"]
                             elif agent_def.agent_type == "tool-calling" and "messages" in agent_node and agent_node["messages"]:
                                 last_msg = agent_node["messages"][-1]
                                 if isinstance(last_msg, AIMessage):
                                      accumulated_final_output = last_msg.content
                                 elif hasattr(last_msg, 'content'):
                                      accumulated_final_output = last_msg.content
                             elif "output" in agent_node: # Fallback
                                 accumulated_final_output = agent_node["output"]
                        
                        # Check for streaming LLM tokens (might be nested differently)
                        # Example: look inside 'agent' node for AIMessageChunk content
                        messages_chunk = agent_node.get("messages") if isinstance(agent_node, dict) else None
                        if messages_chunk and isinstance(messages_chunk, list) and messages_chunk:
                            last_msg_chunk = messages_chunk[-1]
                            if hasattr(last_msg_chunk, 'content'):
                                llm_token = last_msg_chunk.content # This might be a full message or a chunk
                                # If it's a chunk (AIMessageChunk), content is the delta
                                # This heuristic needs testing!
                                if hasattr(last_msg_chunk, '__class__') and 'Chunk' in last_msg_chunk.__class__.__name__:
                                    pass # Content is likely the token chunk
                                else: # It might be a full message, only stream if it's the first time?
                                    # This part is tricky without knowing exact stream format
                                    # Let's assume for now `content` on non-chunk messages isn't streamed token-by-token
                                    llm_token = None 

                    # --- End Example Parsing Logic ---

                    if llm_token:
                         yield f"data: {json.dumps({'type': 'token', 'content': llm_token})}\n\n"
                    # We could yield other event types here (e.g., tool start/end)
                    # yield f"event: tool_start\ndata: {json.dumps({...})}\n\n"

            except Exception as stream_error:
                 import traceback
                 print(f"[AGENT STREAM ERROR] Error during agent stream: {stream_error}\n{traceback.format_exc()}")
                 yield f"event: error\ndata: {json.dumps({'error': str(stream_error)})}\n\n"
                 return # Stop the generator on error

            # After the stream finishes, finalize history and send the end event
            final_ai_message = None
            if accumulated_final_output is not None:
                final_ai_message = AIMessage(content=str(accumulated_final_output))
                if agent_def.has_memory:
                    # Add the final AI message to history
                     if not updated_history_for_stream or updated_history_for_stream[-1] != final_ai_message:
                         updated_history_for_stream.append(final_ai_message)

            # Convert history to JSON serializable format for the final event
            returned_chat_history = []
            if agent_def.has_memory:
                for msg in updated_history_for_stream:
                     if isinstance(msg, HumanMessage):
                         returned_chat_history.append({"role": "user", "content": msg.content})
                     elif isinstance(msg, AIMessage):
                         returned_chat_history.append({"role": "assistant", "content": msg.content})
                     elif isinstance(msg, SystemMessage):
                         returned_chat_history.append({"role": "system", "content": msg.content})
            
            end_data = {
                 "final_output": accumulated_final_output,
                 "chat_history": returned_chat_history
            }
            yield f"event: end\ndata: {json.dumps(end_data)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
        # --- End Streaming Logic ---

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found."
        )
    except ValueError as e:
        print(f"[AGENT ERROR] ValueError in agent '{agent_id}': {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        print(f"[AGENT ERROR] KeyError in agent '{agent_id}': {e}")
        raise HTTPException(status_code=422, detail=f"Missing key: {e}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[AGENT ERROR] Exception in agent '{agent_id}': {e}\n{tb}")
        if 'llm-math' in str(e):
            print("[AGENT ERROR] The error may be due to llm-math tool instantiation or usage.")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=(
                f"Internal error: {e}"
            )
        )

@app.get("/embeddings/nvidia/models")
def list_nvidia_embedding_models(
    base_url: str = Query(None, description="NVIDIA endpoint base URL"),
    api_key: str = Query(None, description="NVIDIA API key (optional, if not set in env)")
):
    """
    Returns the list of available NVIDIA embedding models from the configured endpoint.
    """
    try:
        models = NVIDIAEmbeddings.get_available_models(base_url=base_url, api_key=api_key)
        return [
            {
                "name": getattr(m, "name", str(m)),
                "description": getattr(m, "description", ""),
                "max_input_length": getattr(m, "max_input_length", None),
            }
            for m in models
        ]
    except Exception as e:
        return {"error": str(e)}

@app.post("/embeddings/nvidia")
async def nvidia_embeddings(
    texts: list[str] = Body(..., description="Texts to embed"),
    model: str = Body("NV-Embed-QA", description="NVIDIA embedding model (see /embeddings/nvidia/models)"),
    base_url: str = Body(None, description="NVIDIA endpoint base URL"),
    api_key: str = Body(None, description="NVIDIA API key"),
    truncate: str = Body("NONE", description="Truncation strategy: NONE, START, END"),
    async_mode: bool = Body(False, description="Use async embedding"),
):
    service = NvidiaEmbeddingService(model=model, base_url=base_url, api_key=api_key, truncate=truncate)
    if async_mode:
        embeddings = await service.aembed_documents(texts)
    else:
        embeddings = service.embed_documents(texts)
    return {"embeddings": embeddings}

@app.post("/documents/upload")
async def upload_and_chunk_document(
    file: UploadFile = File(...),
    loader_type: str = None,
    chunking_strategy: str = ChunkingStrategy.RECURSIVE.value,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: dict = None
):
    """
    Upload a document, auto-detect or specify loader, chunk it, and return chunks.
    """
    import tempfile
    import shutil
    if metadata is None:
        metadata = {}
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        docs = load_document(tmp_path, loader_type=loader_type)
        chunker = Chunker(
            strategy=ChunkingStrategy(chunking_strategy),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        all_chunks = []
        for doc in docs:
            doc_meta = metadata.copy()
            doc_meta.update(doc.get("metadata", {}))
            chunks = chunker.chunk_text(doc["text"], metadata=doc_meta)
            all_chunks.extend(chunks)
        return {"chunks": all_chunks}
    finally:
        os.remove(tmp_path)

@app.post("/documents/chunk")
async def chunk_raw_text(
    text: str = Body(..., description="Raw text to chunk"),
    chunking_strategy: str = Body(ChunkingStrategy.RECURSIVE.value),
    chunk_size: int = Body(512),
    chunk_overlap: int = Body(50),
    metadata: dict = Body({}, description="Metadata to attach to each chunk")
):
    """
    Chunk raw text using the specified strategy and return chunks.
    """
    chunker = Chunker(
        strategy=ChunkingStrategy(chunking_strategy),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = chunker.chunk_text(text, metadata=metadata)
    return {"chunks": chunks}

@app.post("/vector/index")
async def vector_index(
    chunks: list = Body(..., description="List of chunks (text + metadata)"),
    store_type: str = Body(
        "faiss", description="Vector store type: faiss or chroma"
    ),
    persist_directory: str = Body(
        None, description="Persist directory for Chroma (optional)"
    )
):
    """
    Index a list of chunks in the specified vector store.
    """
    if not chunks:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks provided for indexing."
        )
    vsm = VectorStoreManager(
        store_type=store_type, persist_directory=persist_directory
    )
    vsm.add_chunks(chunks)
    return {"status": "ok", "num_chunks": len(chunks)}

@app.post("/vector/search")
async def vector_search(
    query: str = Body(..., description="Query string"),
    top_k: int = Body(5, description="Number of results to return"),
    filters: dict = Body({}, description="Metadata filters (optional)"),
    store_type: str = Body(
        "faiss", description="Vector store type: faiss or chroma"
    ),
    persist_directory: str = Body(
        None, description="Persist directory for Chroma (optional)"
    ),
    retriever_type: str = Body("vector", description="Retriever type: vector, multi_query, ensemble, long_context_reorder"),
    rerank: bool = Body(
        False, description="Whether to rerank results using NVIDIA Reranker (optional)"
    ),
    rerank_model: str = Body(
        None, description="NVIDIA rerank model (optional)"
    ),
    rerank_api_key: str = Body(
        None, description="NVIDIA API key for reranker (optional)"
    ),
    rerank_top_n: int = Body(
        None, description="Number of reranked results to return (optional)"
    ),
    rerank_base_url: str = Body(
        None, description="NVIDIA rerank base URL (optional)"
    )
):
    """
    Search the vector store for similar chunks. Supports advanced retrievers and reranking.
    """
    vsm = VectorStoreManager(
        store_type=store_type, persist_directory=persist_directory
    )
    if vsm.vectorstore is None:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vector store is not initialized. Index chunks first."
        )
    results = vsm.search(query, top_k=top_k, filters=filters)
    if retriever_type == "multi_query":
        if vsm.vectorstore is None:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vector store is not initialized. Index chunks first."
            )
        base_retriever = vsm.vectorstore.as_retriever()
        llm = ChatNVIDIA(
            model="meta/llama3-70b-instruct", # Hardcode a default model for now
            api_key=settings.NVIDIA_API_KEY
        )
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )
        docs = multi_query_retriever.get_relevant_documents(query)
        results = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in docs
        ][:top_k]
    elif retriever_type == "ensemble":
        if vsm.vectorstore is None:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vector store is not initialized. Index chunks first."
            )
        vector_retriever = vsm.vectorstore.as_retriever()
        class KeywordRetriever(BaseRetriever):
            vectorstore: Any
            k: int
            def _get_relevant_documents(self, query: str) -> List[Document]:
                 # Simplified: Return empty list for FAISS as keyword search is hard
                 # Avoids the embedding error with empty query
                 print("[WARN] KeywordRetriever for FAISS is not implemented, returning empty list.")
                 return []

        keyword_retriever_instance = KeywordRetriever(vectorstore=vsm.vectorstore, k=top_k)
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever_instance], weights=[0.5, 0.5]
        )
        docs = ensemble.get_relevant_documents(query)
        results = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in docs
        ][:top_k]
    elif retriever_type == "long_context_reorder":
        docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in results]
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs, query=query)
        results = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in reordered_docs
        ][:top_k]
    if rerank and results:
        from langchain_nvidia_ai_endpoints import NVIDIARerank
        rerank_kwargs = {}
        if rerank_model:
            rerank_kwargs["model"] = rerank_model
        if rerank_api_key:
            rerank_kwargs["api_key"] = rerank_api_key
        if rerank_base_url:
            rerank_kwargs["base_url"] = rerank_base_url
        if rerank_top_n:
            rerank_kwargs["top_n"] = rerank_top_n
        else:
            rerank_kwargs["top_n"] = min(top_k, len(results))
        reranker = NVIDIARerank(**rerank_kwargs)
        docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in results]
        reranked_docs = reranker.compress_documents(query=query, documents=docs)
        reranked_results = []
        for doc in reranked_docs:
            orig = next((r for r in results if r["text"] == doc.page_content), {})
            score = orig.get("score")
            reranked_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) if score is not None else None
            })
        return {"results": reranked_results}
    return {"results": results}

@app.post("/vector/hybrid_search")
async def vector_hybrid_search(
    query: str = Body(..., description="Query string for vector search"),
    keyword: str = Body(
        None, description="Keyword for keyword search (optional)"
    ),
    top_k: int = Body(5, description="Number of results to return"),
    filters: dict = Body({}, description="Metadata filters (optional)"),
    store_type: str = Body(
        "faiss", description="Vector store type: faiss or chroma"
    ),
    persist_directory: str = Body(
        None, description="Persist directory for Chroma (optional)"
    ),
    retriever_type: str = Body("hybrid", description="Retriever type: hybrid, multi_query, ensemble, long_context_reorder"),
    rerank: bool = Body(
        False, description="Whether to rerank results using NVIDIA Reranker (optional)"
    ),
    rerank_model: str = Body(
        None, description="NVIDIA rerank model (optional)"
    ),
    rerank_api_key: str = Body(
        None, description="NVIDIA API key for reranker (optional)"
    ),
    rerank_top_n: int = Body(
        None, description="Number of reranked results to return (optional)"
    ),
    rerank_base_url: str = Body(
        None, description="NVIDIA rerank base URL (optional)"
    )
):
    """
    Hybrid search: combines vector similarity and keyword filtering. Supports advanced retrievers and reranking.
    """
    vsm = VectorStoreManager(
        store_type=store_type, persist_directory=persist_directory
    )
    if vsm.vectorstore is None:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vector store is not initialized. Index chunks first."
        )
    results = vsm.hybrid_search(
        query=query, keyword=keyword, top_k=top_k, filters=filters
    )
    if retriever_type == "multi_query":
        if vsm.vectorstore is None:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vector store is not initialized. Index chunks first."
            )
        base_retriever = vsm.vectorstore.as_retriever()
        llm = ChatNVIDIA(
            model="meta/llama3-70b-instruct", # Hardcode a default model for now
            api_key=settings.NVIDIA_API_KEY
        )
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )
        docs = multi_query_retriever.get_relevant_documents(query)
        results = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in docs
        ][:top_k]
    elif retriever_type == "ensemble":
        if vsm.vectorstore is None:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vector store is not initialized. Index chunks first."
            )
        vector_retriever = vsm.vectorstore.as_retriever()
        class KeywordRetriever(BaseRetriever):
            vectorstore: Any
            k: int
            def _get_relevant_documents(self, query: str) -> List[Document]:
                 # Simplified: Return empty list for FAISS as keyword search is hard
                 print("[WARN] KeywordRetriever for FAISS is not implemented, returning empty list.")
                 return []

        keyword_retriever_instance = KeywordRetriever(vectorstore=vsm.vectorstore, k=top_k)
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever_instance], weights=[0.5, 0.5]
        )
        docs = ensemble.get_relevant_documents(query)
        results = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in docs
        ][:top_k]
    elif retriever_type == "long_context_reorder":
        docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in results]
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs, query=query)
        results = [
            {"text": doc.page_content, "metadata": doc.metadata} for doc in reordered_docs
        ][:top_k]
    if rerank and results:
        from langchain_nvidia_ai_endpoints import NVIDIARerank
        rerank_kwargs = {}
        if rerank_model:
            rerank_kwargs["model"] = rerank_model
        if rerank_api_key:
            rerank_kwargs["api_key"] = rerank_api_key
        if rerank_base_url:
            rerank_kwargs["base_url"] = rerank_base_url
        if rerank_top_n:
            rerank_kwargs["top_n"] = rerank_top_n
        else:
            rerank_kwargs["top_n"] = min(top_k, len(results))
        reranker = NVIDIARerank(**rerank_kwargs)
        docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in results]
        reranked_docs = reranker.compress_documents(query=query, documents=docs)
        reranked_results = []
        for doc in reranked_docs:
            orig = next((r for r in results if r["text"] == doc.page_content), {})
            score = orig.get("score")
            reranked_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) if score is not None else None
            })
        return {"results": reranked_results}
    return {"results": results}

@app.post("/vector/delete")
async def vector_delete(
    store_type: str = Body("chroma", description="Vector store type (must be chroma)"),
    persist_directory: str = Body(None, description="Persist directory for Chroma (optional)"),
    filter: dict = Body(None, description="Metadata filter for deletion (optional)"),
    ids: list = Body(None, description="List of document IDs to delete (optional)")
):
    """
    Delete chunks by metadata filter or document IDs. Only supported for Chroma.
    """
    vsm = VectorStoreManager(store_type=store_type, persist_directory=persist_directory)
    try:
        vsm.delete_chunks(filter=filter, ids=ids)
        return {"status": "ok"}
    except NotImplementedError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/vector/update")
async def vector_update(
    store_type: str = Body("chroma", description="Vector store type (must be chroma)"),
    persist_directory: str = Body(None, description="Persist directory for Chroma (optional)"),
    doc_id: str = Body(..., description="Document ID to update"),
    new_text: str = Body(None, description="New text for the chunk (optional)"),
    new_metadata: dict = Body(None, description="New metadata for the chunk (optional)")
):
    """
    Update a chunk by document ID. Only supported for Chroma.
    """
    vsm = VectorStoreManager(store_type=store_type, persist_directory=persist_directory)
    try:
        vsm.update_chunk(doc_id=doc_id, new_text=new_text, new_metadata=new_metadata)
        return {"status": "ok"}
    except NotImplementedError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
