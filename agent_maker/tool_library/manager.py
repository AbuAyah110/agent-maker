import os
import json
# import yaml # Not used yet, will be for custom tool loading
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import requests

# --- Pydantic Models for Tools ---


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any | None = None


class ToolMetadata(BaseModel):
    version: str = "1.0.0"
    author: str | None = None
    tags: List[str] = []
    # For LangChain tools, we might include the module it comes from
    langchain_module: Optional[str] = None


class LangChainToolData(BaseModel):
    name: str = Field(
        ...,
        description="Unique name/identifier for the LangChain tool."
    )
    description: str = Field(..., description="What the tool does.")
    # How to instantiate or use it, can be a class name or a function name
    usage_class_or_function: str
    parameters: List[ToolParameter] = []  # Key parameters
    metadata: ToolMetadata = Field(default_factory=ToolMetadata)


class CustomToolData(BaseModel):
    name: str = Field(
        ...,
        description="Unique name/identifier for the custom tool."
    )
    description: str = Field(..., description="What the tool does.")
    script_path: str = Field(  # Path to the Python script
        ...,
        description="Path to the Python script implementing the tool."
    )
    parameters: List[ToolParameter] = []
    metadata: ToolMetadata = ToolMetadata()


# --- Predefined LangChain Tools (Curated List) ---

# This is a simplified, curated list. In a real system, this could be
# more extensive and potentially loaded from a configuration file or a
# more sophisticated discovery mechanism.
PREDEFINED_LANGCHAIN_TOOLS: List[LangChainToolData] = [
    LangChainToolData(
        name="llm-math",
        description=(
            "A tool that uses an LLM to do math. Good for word problems and "
            "chain-of-thought math."
        ),
        usage_class_or_function="LLMMathChain",  # From langchain.chains
        parameters=[
            ToolParameter(
                name="llm",
                type="BaseLanguageModel",
                description="The language model to use.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain.chains",
            tags=["math", "llm-powered"]
        )
    ),
    LangChainToolData(
        name="wikipedia",
        description=(
            "A tool that connects to Wikipedia to search for and retrieve "
            "articles."
        ),
        usage_class_or_function="WikipediaQueryRun",  # From langchain_community.tools
        parameters=[
            ToolParameter(
                name="api_wrapper",
                type="WikipediaAPIWrapper",
                description="Instance of WikipediaAPIWrapper.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.tools",
            tags=["search", "knowledge-base"]
        )
    ),
    LangChainToolData(
        name="searx-search",
        description=(
            "A tool that uses a Searx instance to perform web searches."
        ),
        usage_class_or_function="SearxSearchRun",  # From langchain_community.tools
        parameters=[
            ToolParameter(
                name="wrapper",
                type="SearxSearchWrapper",
                description=(
                    "Instance of SearxSearchWrapper configured with the Searx "
                    "host."
                ),
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.tools",
            tags=["search", "web", "searx"]
        )
    ),
    LangChainToolData(
        name="tavily_search",
        description=(
            "Tool for interacting with the Tavily Search API. Useful for "
            "current event information or broad knowledge queries. Requires "
            "TAVILY_API_KEY."
        ),
        usage_class_or_function="TavilySearchResults",
        parameters=[
            ToolParameter(
                name="max_results",
                type="int",
                description="Maximum number of search results to return.",
                required=False,
                default=5
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_tavily.tools",
            tags=["search", "web", "tavily"]
        )
    ),
    LangChainToolData(
        name="github_toolkit",
        description=(
            "Toolkit for interacting with GitHub. Provides tools for managing "
            "issues, pull requests, files, and searching code. Requires "
            "GitHub API configuration."
        ),
        usage_class_or_function="GitHubToolkit",  # from_github_api_wrapper
        parameters=[
            ToolParameter(
                name="gh_api_wrapper",
                type="GitHubAPIWrapper",
                description=(
                    "Instance of GitHubAPIWrapper configured with credentials."
                ),
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.agent_toolkits.github.toolkit"
            ),
            tags=["version-control", "code", "issues", "github"]
        )
    ),
    LangChainToolData(
        name="gitlab_toolkit",
        description=(
            "Toolkit for interacting with GitLab. Allows managing issues, merge "
            "requests, repositories, comments, and searching projects. "
            "Requires GitLab API configuration."
        ),
        usage_class_or_function="GitLabToolkit",  # from_gitlab_api_wrapper
        parameters=[
            ToolParameter(
                name="gitlab_api_wrapper",
                type="GitLabAPIWrapper",
                description=(
                    "Instance of GitLabAPIWrapper configured with URL/token."
                ),
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.agent_toolkits.gitlab.toolkit"
            ),
            tags=["version-control", "code", "issues", "gitlab"]
        )
    ),
    LangChainToolData(
        name="gmail_toolkit",
        description=(
            "Toolkit for interacting with the Gmail API. Provides tools to "
            "search, read, create, and send emails. Requires Gmail API "
            "credentials."
        ),
        usage_class_or_function="GmailToolkit",
        parameters=[
            ToolParameter(
                name="api_resource",
                type="googleapiclient.discovery.Resource",
                description=(
                    "The authenticated Gmail API resource service object."
                ),
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_google_community.agent_toolkits.gmail.toolkit"
            ),
            tags=["email", "communication", "google", "gmail"]
        )
    ),
    LangChainToolData(
        name="jira_toolkit",
        description=(
            "Toolkit for interacting with a Jira instance. Allows searching "
            "for issues (JQL), creating issues, fetching projects, and other "
            "Jira API actions. Requires Jira API credentials and instance URL."
        ),
        usage_class_or_function="JiraToolkit",  # from_jira_api_wrapper
        parameters=[
            ToolParameter(
                name="jira_api_wrapper",
                type="JiraAPIWrapper",
                description=(
                    "Instance of JiraAPIWrapper configured with URL and "
                    "credentials."
                ),
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.agent_toolkits.jira.toolkit",
            tags=["project-management", "issue-tracking", "jira", "atlassian"]
        )
    ),
    LangChainToolData(
        name="office365_toolkit",
        description=(
            "Toolkit for interacting with Office 365 (Microsoft 365). Allows "
            "searching emails/events, sending emails/invites, creating "
            "drafts. Requires Microsoft Graph API credentials (Client "
            "ID/Secret)."
        ),
        usage_class_or_function="O365Toolkit",
        parameters=[  # Instantiation might rely on env vars or O365 lib config
            ToolParameter(
                name="account",
                type="O365.Account",
                description="Optional pre-configured O365 Account object.",
                required=False
            )
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.agent_toolkits.office365.toolkit"
            ),
            tags=[
                "email", "calendar", "microsoft", "office365", "communication"
            ]
        )
    ),
    LangChainToolData(
        name="slack_toolkit",
        description=(
            "Toolkit for interacting with Slack. Allows getting channel info, "
            "getting messages, scheduling messages, and sending messages. "
            "Requires SLACK_USER_TOKEN environment variable."
        ),
        usage_class_or_function="SlackToolkit",
        parameters=[  # Instantiation relies on env vars or slack_sdk config
            ToolParameter(
                name="client",
                type="slack_sdk.WebClient",
                description="Optional pre-configured Slack WebClient object.",
                required=False
            )
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.agent_toolkits.slack.toolkit"
            ),
            tags=["communication", "messaging", "collaboration", "slack"]
        )
    ),
    LangChainToolData(
        name="twilio",
        description=(
            "A tool that uses the Twilio API to send messages via SMS or "
            "Twilio Messaging Channels (e.g., WhatsApp). Requires Twilio "
            "Account SID, Auth Token, and a 'From' number."
        ),
        usage_class_or_function="TwilioAPIWrapper",
        parameters=[
            ToolParameter(
                name="account_sid",
                type="str",
                description=(
                    "Twilio Account SID. Can be set via env "
                    "TWILIO_ACCOUNT_SID."
                ),
                required=False  # If env var is set
            ),
            ToolParameter(
                name="auth_token",
                type="str",
                description=(
                    "Twilio Auth Token. Can be set via env TWILIO_AUTH_TOKEN."
                ),
                required=False  # If env var is set
            ),
            ToolParameter(
                name="from_number",
                type="str",
                description=(
                    "Twilio phone number to send messages from. Can be set via "
                    "env TWILIO_FROM_NUMBER."
                ),
                required=False  # If env var is set
            ),
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.utilities.twilio",
            tags=["communication", "messaging", "sms", "whatsapp", "twilio"]
        )
    ),
    LangChainToolData(
        name="playwright_browser_toolkit",
        description=(
            "Toolkit for controlling a Playwright browser. Includes tools for "
            "navigation (navigate, back), interaction (click), and data "
            "extraction (text, hyperlinks, elements). Requires an async "
            "browser instance."
        ),
        usage_class_or_function="PlayWrightBrowserToolkit",  # Instantiated with from_browser
        parameters=[
            ToolParameter(
                name="async_browser",
                type="playwright.async_api.Browser",
                description=(
                    "An instance of an async Playwright Browser, typically "
                    "created via create_async_playwright_browser."
                ),
                required=True
            )
            # Individual tools within the toolkit have their own parameters
            # (e.g., url, selector, attributes)
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.agent_toolkits.playwright.toolkit"
            ),
            tags=["browser", "web-automation", "scraping", "playwright"]
        )
    ),
    LangChainToolData(
        name="requests_toolkit",
        description=(
            "Toolkit for making REST requests (GET, POST, PATCH, PUT, DELETE). "
            "Requires 'allow_dangerous_requests=True' for security opt-in."
        ),
        # Note: This toolkit returns individual tools (GET, POST, etc.)
        # via get_tools()
        usage_class_or_function="RequestsToolkit",
        parameters=[
            ToolParameter(
                name="requests_wrapper",
                type="TextRequestsWrapper | JsonRequestsWrapper",
                description=(
                    "Wrapper for executing requests (e.g., "
                    "TextRequestsWrapper())."
                ),
                required=True
            ),
            ToolParameter(
                name="allow_dangerous_requests",
                type="bool",
                description=(
                    "Must be set to True to enable request execution."
                ),
                required=False,  # Technically False, but needs to be True to work
                default=False
            ),
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.agent_toolkits.openapi.toolkit"
            ),
            tags=["http", "rest", "api", "requests"]
        )
    ),
    LangChainToolData(
        name="sql_database_toolkit",
        description=(
            "Toolkit for interacting with SQL databases. Allows querying "
            "database schema, checking SQL queries, and executing SQL queries. "
            "Requires a SQLDatabase object (from a SQLAlchemy engine) and an "
            "LLM for query checking."
        ),
        usage_class_or_function="SQLDatabaseToolkit",
        parameters=[
            ToolParameter(
                name="db",
                type="langchain_community.utilities.sql_database.SQLDatabase",
                description=(
                    "An instance of SQLDatabase, connected to the target "
                    "database."
                ),
                required=True
            ),
            ToolParameter(
                name="llm",
                type="langchain_core.language_models.base.BaseLanguageModel",
                description="A language model for the query checker tool.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.agent_toolkits.sql.toolkit",
            tags=["database", "sql", "query", "structured-data"]
        )
    ),
    LangChainToolData(
        name="spark_sql_toolkit",
        description=(
            "Toolkit for interacting with Spark SQL. Allows listing tables, "
            "getting schema, and running Spark SQL queries. Requires a "
            "SparkSQL object (from a Spark session) and an LLM for query "
            "checking."
        ),
        usage_class_or_function="SparkSQLToolkit",
        parameters=[
            ToolParameter(
                name="db",
                type="langchain_community.utilities.spark_sql.SparkSQL",
                description=(
                    "An instance of SparkSQL, connected to the Spark session."
                ),
                required=True
            ),
            ToolParameter(
                name="llm",
                type="langchain_core.language_models.base.BaseLanguageModel",
                description="A language model for the query checker tool.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.agent_toolkits.spark_sql.toolkit"
            ),
            tags=["database", "sql", "spark", "big-data", "query"]
        )
    ),
    LangChainToolData(
        name="ads4gpts_toolkit",
        description=(
            "Toolkit for integrating AI-native advertising (ADS4GPTs). "
            "Includes tools for generating inline sponsored responses and "
            "suggested ad prompts based on user context. Requires "
            "ADS4GPTS_API_KEY."
        ),
        usage_class_or_function="Ads4gptsToolkit",
        parameters=[
            ToolParameter(
                name="ads4gpts_api_key",
                type="str",
                description="API key for the ADS4GPTs service.",
                required=False  # Can be set via env var
            )
        ],
        metadata=ToolMetadata(
            langchain_module="ads4gpts_langchain",
            tags=["advertising", "monetization", "ai-ads"]
        )
    ),
    LangChainToolData(
        name="agentql_extract_web_data_tool",
        description=(
            "Extracts structured data as JSON from a public web page URL "
            "using either an AgentQL query or a Natural Language "
            "description. Requires AGENTQL_API_KEY."
        ),
        usage_class_or_function="ExtractWebDataTool",
        parameters=[
            ToolParameter(
                name="url",
                type="str",
                description=(
                    "The URL of the public web page to extract data from."
                ),
                required=True
            ),
            ToolParameter(
                name="query",
                type="str",
                description="AgentQL query for precise data extraction.",
                required=False  # Either query or prompt is required
            ),
            ToolParameter(
                name="prompt",
                type="str",
                description="Natural language description of data to extract.",
                required=False  # Either query or prompt is required
            ),
            ToolParameter(
                name="api_key",
                type="str",
                description=(
                    "AgentQL API key (optional if AGENTQL_API_KEY env var "
                    "set)."
                ),
                required=False
            ),
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_agentql.tools",
            tags=[
                "web-scraping", "data-extraction", "structured-data",
                "agentql", "rest-api"
            ]
        )
    ),
    LangChainToolData(
        name="agentql_browser_toolkit",
        description=(
            "Toolkit for browser interaction using AgentQL. Extracts "
            "structured data or finds web elements using natural language or "
            "AgentQL queries within an active Playwright browser session. "
            "Requires AGENTQL_API_KEY."
        ),
        usage_class_or_function="AgentQLBrowserToolkit",
        parameters=[
            ToolParameter(
                name="async_browser",
                type="playwright.async_api.Browser",
                description="An active async Playwright browser instance.",
                required=True
            ),
            ToolParameter(
                name="api_key",
                type="str",
                description=(
                    "AgentQL API key (optional if AGENTQL_API_KEY env var "
                    "set)."
                ),
                required=False
            ),
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_agentql",
            tags=[
                "browser", "web-automation", "web-scraping",
                "data-extraction", "agentql"
            ]
        )
    ),
    LangChainToolData(
        name="apify_actor",
        description=(
            "Runs a specified Apify Actor with the given input and returns "
            "its results. Useful for various web scraping, crawling, and "
            "data extraction tasks. Requires APIFY_API_TOKEN."
        ),
        usage_class_or_function="ApifyActorsTool",
        parameters=[
            ToolParameter(
                name="actor_id",
                type="str",
                description=(
                    "The ID of the Apify Actor to run (e.g., "
                    "'apify/website-content-crawler')."
                ),
                required=True
            ),
            ToolParameter(
                name="run_input",
                type="dict",
                description=(
                    "Dictionary representing the input for the actor run."
                ),
                required=True
            ),
            ToolParameter(
                name="api_token",
                type="str",
                description=(
                    "Apify API token (optional if APIFY_API_TOKEN env var "
                    "set)."
                ),
                required=False
            ),
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_apify",
            tags=["web-scraping", "automation", "apify"]
        )
    ),
    LangChainToolData(
        name="arxiv",
        description=(
            "Searches Arxiv.org for scientific articles. Useful for questions "
            "about Physics, Math, CS, Biology, Finance, Statistics, "
            "Electrical Engineering, and Economics. Input should be a search "
            "query or Arxiv ID."
        ),
        usage_class_or_function="ArxivQueryRun",
        parameters=[
            ToolParameter(
                name="query",
                type="str",
                description="The search query or ArXiv ID.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.tools.arxiv.tool",
            tags=["research", "science", "academic", "papers", "arxiv"]
        )
    ),
    LangChainToolData(
        name="shell_tool",
        description=(
            "Runs shell commands on the local machine. Can execute a single "
            "command or a list of commands. Use with caution."
        ),
        usage_class_or_function="ShellTool",
        parameters=[
            ToolParameter(
                name="commands",
                type="Union[str, List[str]]",  # Type hint for documentation
                description=(
                    "A command string or a list of command strings to execute."
                ),
                required=True
            ),
            ToolParameter(
                name="ask_human_input",
                type="bool",
                description="If True, prompt user before executing commands.",
                required=False,
                default=False
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.tools.shell.tool",
            tags=["shell", "terminal", "os", "command-line"]
        )
    ),
    LangChainToolData(
        name="dataforseo_toolkit",
        description=(
            "Toolkit for accessing DataForSEO APIs. Provides various SEO tools "
            "like SERP data, keyword research, on-page analysis, etc. "
            "Requires DataForSEO API credentials."
        ),
        usage_class_or_function="DataForSeoAPIWrapper",  # Core wrapper
        parameters=[
            ToolParameter(
                name="a_key",
                type="str",
                description="DataForSEO API key.",
                required=False  # Can be set via DATAFORSEO_API_KEY
            ),
            ToolParameter(
                name="api_login",
                type="str",
                description="DataForSEO API login.",
                required=False  # Can be set via DATAFORSEO_LOGIN
            ),
            ToolParameter(
                name="api_password",
                type="str",
                description="DataForSEO API password.",
                required=False  # Can be set via DATAFORSEO_PASSWORD
            )
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.utilities.dataforseo_api_wrapper"
            ),
            tags=["seo", "marketing", "serp", "keywords", "dataforseo"]
        )
    ),
    LangChainToolData(
        name="discord_webhook_tool",
        description=(
            "Sends a message to a Discord channel via a webhook. Requires a "
            "Discord webhook URL (WEBHOOK_URL) and can optionally use a "
            "custom username (WEBHOOK_USERNAME) set as environment variables."
        ),
        usage_class_or_function="DiscordWebhookTool",
        parameters=[
            ToolParameter(
                name="message",
                type="str",
                description="The message content to send to Discord.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_discord",
            tags=["communication", "messaging", "discord", "webhook"]
        )
    ),
    LangChainToolData(
        name="eleven_labs_text2speech",
        description=(
            "A tool to convert text to speech using ElevenLabs. Requires an "
            "ElevenLabs API key (ELEVENLABS_API_KEY env var). Optional "
            "parameters for model and voice ID can be configured during setup."
        ),
        usage_class_or_function="ElevenLabsText2SpeechTool",
        parameters=[
            ToolParameter(
                name="query",
                type="str",
                description="The text to convert to speech.",
                required=True
            )
            # Constructor args handled separately
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.tools.eleven_labs.text2speech"
            ),
            tags=["audio", "speech", "tts", "elevenlabs"]
        )
    ),
    LangChainToolData(
        name="file_management_toolkit",
        description=(
            "Toolkit for interacting with the local file system. Includes "
            "tools for reading, writing, listing, copying, moving, and "
            "deleting files/dirs. Use with caution and preferably with a "
            "restricted `root_dir`."
        ),
        usage_class_or_function="FileManagementToolkit",
        parameters=[
            ToolParameter(
                name="root_dir",
                type="str",
                description=(
                    "The root directory for file operations. If not set, "
                    "defaults to current working directory (use with extreme "
                    "caution)."
                ),
                required=False
            ),
            ToolParameter(
                name="selected_tools",
                type="List[str]",
                description=(
                    "Optional list of specific tool names to include (e.g., "
                    "['read_file', 'list_directory'])."
                ),
                required=False
            )
            # Individual tools provided by the toolkit have their own
            # specific args (e.g., file_path, text, dir_path, pattern)
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.agent_toolkits",
            tags=["filesystem", "file-management", "os", "local"]
        )
    ),
    LangChainToolData(
        name="financial_datasets_toolkit",
        description=(
            "Toolkit for accessing financial data (balance sheets, income "
            "statements, cash flow statements) for publicly traded "
            "companies. Requires FINANCIAL_DATASETS_API_KEY."
        ),
        usage_class_or_function="FinancialDatasetsToolkit",
        parameters=[
            ToolParameter(
                name="api_wrapper",
                type="FinancialDatasetsAPIWrapper",
                description=(
                    "Instance of FinancialDatasetsAPIWrapper configured with "
                    "API key."
                ),
                required=True
            )
            # Specific tools (BalanceSheets, etc.) take args like ticker, period
        ],
        metadata=ToolMetadata(
            langchain_module=(
                "langchain_community.agent_toolkits.financial_datasets.toolkit"
            ),
            tags=["finance", "stocks", "company-data", "financial-statements"]
        )
    ),
    LangChainToolData(
        name="google_drive_search",
        description=(
            "Searches for files and documents within Google Drive based on a "
            "query. Can search within a specific folder or across the entire "
            "drive. Requires Google API credentials."
        ),
        usage_class_or_function="GoogleDriveSearchTool",
        parameters=[
            ToolParameter(
                name="api_wrapper",
                type="GoogleDriveAPIWrapper",
                description=(
                    "Instance of GoogleDriveAPIWrapper configured with "
                    "credentials, folder_id (optional), and other search "
                    "parameters."
                ),
                required=True
            )
            # The run method takes a 'query' string.
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_googledrive.tools.google_drive.tool",
            tags=["file-search", "cloud-storage", "google-drive", "documents"]
        )
    ),
    LangChainToolData(
        name="google_finance",
        description=(
            "Fetches real-time stock quotes, market trends, and financial "
            "news from Google Finance via SerpApi. Requires SERPAPI_API_KEY."
        ),
        usage_class_or_function="GoogleFinanceQueryRun",
        parameters=[
            ToolParameter(
                name="api_wrapper",
                type="GoogleFinanceAPIWrapper",
                description=(
                    "Instance of GoogleFinanceAPIWrapper configured with "
                    "SerpApi key."
                ),
                required=True
            )
            # The run method takes a 'query' (stock ticker/company name).
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.tools.google_finance.tool",
            tags=["finance", "stocks", "market-data", "news", "google-finance"]
        )
    ),
    LangChainToolData(
        name="google_scholar",
        description=(
            "Searches Google Scholar for academic papers, citations, and "
            "authors. Requires SERPAPI_API_KEY."
        ),
        usage_class_or_function="GoogleScholarQueryRun",
        parameters=[
            ToolParameter(
                name="query",
                type="str",
                description="The search query for Google Scholar.",
                required=True
            )
            # The GoogleScholarAPIWrapper is implicitly used by the tool
        ],
        metadata=ToolMetadata(
            langchain_module="langchain_community.tools.google_scholar",
            tags=[
                "research", "academic", "papers", "citations", "google",
                "serpapi"
            ]
        )
    )
    # Add next tool here
]

# --- Custom Tool Placeholders ---


# In a real implementation, this would scan a directory or load from a config
CUSTOM_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "tools")
# Placeholder: Load custom tool data (e.g., from JSON/YAML files in tools/)
# Example (if tools/my_custom_tool.json existed):
# { "name": "my_custom_tool", "description": ... }


class ToolNotFoundError(Exception):
    """Exception raised when a specific tool cannot be found."""
    pass

# --- Manager Functions ---


def list_langchain_tools() -> List[LangChainToolData]:
    """Lists all predefined LangChain tools available in this manager."""
    return PREDEFINED_LANGCHAIN_TOOLS


def get_langchain_tool_details(name: str) -> Optional[LangChainToolData]:
    """Gets the detailed information for a specific predefined LangChain tool."""
    for tool in PREDEFINED_LANGCHAIN_TOOLS:
        if tool.name == name:
            return tool
    return None


def list_custom_tools() -> List[Dict[str, Any]]:
    """Lists all custom tools found (placeholder)."""
    # Placeholder: In a real implementation, scan CUSTOM_TOOLS_DIR for
    # .json/.yaml and parse them into CustomToolData or similar structure.
    print(f"Scanning for custom tools in: {CUSTOM_TOOLS_DIR}")
    # Example return (replace with actual scanning logic later):
    return [
        # {"name": "example_custom_tool", "description": "Does something custom"}
    ]


def load_custom_tool(name: str) -> Optional[Dict[str, Any]]:
    """Loads the definition of a specific custom tool (placeholder)."""
    # Placeholder: Load and parse the specific tool's definition file
    print(f"Attempting to load custom tool: {name}")
    # Example (if tools/my_custom_tool.json existed and matched 'name'):
    # tool_path = os.path.join(CUSTOM_TOOLS_DIR, f"{name}.json")
    # if os.path.exists(tool_path):
    #     with open(tool_path, 'r') as f:
    #         return json.load(f)
    return None


def save_custom_tool(tool_data: CustomToolData):
    """Saves a custom tool definition (placeholder)."""
    # Placeholder: Save the tool_data (likely as JSON/YAML) into
    # CUSTOM_TOOLS_DIR
    file_path = os.path.join(
        CUSTOM_TOOLS_DIR, f"{tool_data.name}.json"
    )  # Assume JSON
    print(f"Saving custom tool definition to: {file_path}")
    try:
        with open(file_path, 'w') as f:
            json.dump(tool_data.model_dump(), f, indent=2)
        print(f"Successfully saved custom tool '{tool_data.name}'.")
    except IOError as e:
        print(f"Error saving custom tool '{tool_data.name}': {e}")
        raise

# --- Example Usage (for testing within this module) ---


if __name__ == "__main__":
    print("--- Predefined LangChain Tools ---")
    lc_tools = list_langchain_tools()
    if lc_tools:
        print(f"Found {len(lc_tools)} predefined LangChain tools:")
        for tool in lc_tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")
            # Safely access metadata attributes
            if hasattr(tool.metadata, "langchain_module") and tool.metadata.langchain_module:
                print(f"    Module: {tool.metadata.langchain_module}")
    else:
        print("No predefined LangChain tools found.")

    print("\nFetching details for 'wikipedia':")
    wiki_tool = get_langchain_tool_details("wikipedia")
    if wiki_tool:
        print(json.dumps(wiki_tool.model_dump(), indent=2))
    else:
        print("Wikipedia tool not found.")

    print("\nFetching details for 'non_existent_tool':")
    non_tool = get_langchain_tool_details("non_existent_tool")
    print(f"Found: {non_tool}")

    print("\n--- Custom Tools (Placeholders) ---")
    custom_tools_list = list_custom_tools()
    print(f"Custom tools found (placeholder): {custom_tools_list}")

    # --- Saving Custom MCP Tool Definitions ---
    print("\n--- Saving Custom MCP Filesystem Tool Definitions ---")

    # 1. read_file
    read_file_tool = CustomToolData(
        name="mcp_filesystem_read_file",
        description="Read complete contents of a file using the Filesystem MCP server.",
        script_path="MCP Server - Filesystem", # Placeholder
        parameters=[
            ToolParameter(name="path", type="string", description="Path to the file to read.", required=True)
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "file-io", "read"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(read_file_tool)
    except Exception as e:
        print(f"Failed to save read_file_tool: {e}")

    # 2. write_file
    write_file_tool = CustomToolData(
        name="mcp_filesystem_write_file",
        description="Create a new file or overwrite an existing one using the Filesystem MCP server. Use with caution.",
        script_path="MCP Server - Filesystem", # Placeholder
        parameters=[
            ToolParameter(name="path", type="string", description="Path where the file should be written.", required=True),
            ToolParameter(name="content", type="string", description="The content to write into the file.", required=True)
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "file-io", "write"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(write_file_tool)
    except Exception as e:
        print(f"Failed to save write_file_tool: {e}")

    # 3. list_directory
    list_dir_tool = CustomToolData(
        name="mcp_filesystem_list_directory",
        description="List the contents of a directory using the Filesystem MCP server.",
        script_path="MCP Server - Filesystem", # Placeholder
        parameters=[
            ToolParameter(name="path", type="string", description="Path to the directory to list.", required=True)
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "directory", "list"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(list_dir_tool)
    except Exception as e:
        print(f"Failed to save list_dir_tool: {e}")

    # 4. read_multiple_files
    read_multi_tool = CustomToolData(
        name="mcp_filesystem_read_multiple_files",
        description="Read multiple files simultaneously using the Filesystem MCP server.",
        script_path="MCP Server - Filesystem",  # Placeholder
        parameters=[
            ToolParameter(
                name="paths",
                type="string[]",
                description="Array of paths to the files to read.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "file-io", "read", "batch"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(read_multi_tool)
    except Exception as e:
        print(f"Failed to save read_multi_tool: {e}")

    # 5. create_directory
    create_dir_tool = CustomToolData(
        name="mcp_filesystem_create_directory",
        description="Create a new directory or ensure it exists using the Filesystem MCP server.",
        script_path="MCP Server - Filesystem",  # Placeholder
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path where the directory should be created.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "directory", "create"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(create_dir_tool)
    except Exception as e:
        print(f"Failed to save create_dir_tool: {e}")

    # 6. move_file
    move_file_tool = CustomToolData(
        name="mcp_filesystem_move_file",
        description="Move or rename files and directories using the Filesystem MCP server.",
        script_path="MCP Server - Filesystem",  # Placeholder
        parameters=[
            ToolParameter(
                name="source",
                type="string",
                description="The source path of the file or directory to move.",
                required=True
            ),
            ToolParameter(
                name="destination",
                type="string",
                description="The destination path.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "file-io", "directory", "move", "rename"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(move_file_tool)
    except Exception as e:
        print(f"Failed to save move_file_tool: {e}")

    # 7. search_files
    search_files_tool = CustomToolData(
        name="mcp_filesystem_search_files",
        description="Recursively search for files/directories using the Filesystem MCP server.",
        script_path="MCP Server - Filesystem",  # Placeholder
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Starting directory for the search.",
                required=True
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="Search pattern (case-insensitive).",
                required=True
            )
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "search", "find"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(search_files_tool)
    except Exception as e:
        print(f"Failed to save search_files_tool: {e}")

    # 8. get_file_info
    get_info_tool = CustomToolData(
        name="mcp_filesystem_get_file_info",
        description="Get detailed file/directory metadata using the Filesystem MCP server.",
        script_path="MCP Server - Filesystem",  # Placeholder
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file or directory.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "metadata", "info"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(get_info_tool)
    except Exception as e:
        print(f"Failed to save get_info_tool: {e}")

    # 9. tree
    tree_tool = CustomToolData(
        name="mcp_filesystem_tree",
        description="Get a hierarchical JSON representation of a directory structure using the Filesystem MCP server.",
        script_path="MCP Server - Filesystem",  # Placeholder
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Directory to traverse.",
                required=True
            ),
            ToolParameter(
                name="depth",
                type="number",
                description="Maximum depth to traverse.",
                required=False,
                default=3
            ),
            ToolParameter(
                name="follow_symlinks",
                type="boolean",
                description="Whether to follow symbolic links.",
                required=False,
                default=False
            )
        ],
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "directory", "structure", "tree"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(tree_tool)
    except Exception as e:
        print(f"Failed to save tree_tool: {e}")

    # 10. list_allowed_directories
    list_allowed_tool = CustomToolData(
        name="mcp_filesystem_list_allowed_directories",
        description="List all directories the Filesystem MCP server is allowed to access.",
        script_path="MCP Server - Filesystem",  # Placeholder
        parameters=[],  # No input parameters
        metadata=ToolMetadata(
            tags=["mcp", "filesystem", "config", "security", "allowed-dirs"],
            author="modelcontextprotocol / mark3labs"
        )
    )
    try:
        save_custom_tool(list_allowed_tool)
    except Exception as e:
        print(f"Failed to save list_allowed_tool: {e}")

    # --- Saving Custom MCP Everything Tool Definitions ---
    print("\n--- Saving Custom MCP Everything Tool Definitions ---")

    # 1. echo (from everything server)
    echo_tool = CustomToolData(
        name="mcp_everything_echo",
        description="A simple test tool from the MCP 'everything' reference server that echoes back its input.",
        script_path="MCP Server - Everything",  # Placeholder
        parameters=[
            ToolParameter(
                name="message",
                type="string",
                description="The message to echo back.",
                required=True
            )
        ],
        metadata=ToolMetadata(
            tags=["mcp", "everything", "test", "echo"],
            author="modelcontextprotocol"
        )
    )
    try:
        save_custom_tool(echo_tool)
    except Exception as e:
        print(f"Failed to save echo_tool: {e}")

    # --- Saving Custom MCP Memory Tool Definitions ---
    print("\n--- Saving Custom MCP Memory Tool Definitions ---")

    # Based on the standard MCP memory tools available
    common_metadata = ToolMetadata(
        tags=["mcp", "memory", "knowledge-graph"],
        author="modelcontextprotocol"
    )
    common_script_path = "MCP Server - Memory" # Placeholder

    # 1. create_entities
    create_entities_tool = CustomToolData(
        name="mcp_memory_create_entities",
        description="Create multiple new entities in the knowledge graph.",
        script_path=common_script_path,
        parameters=[
            ToolParameter(
                name="entities",
                type="List[dict]", # [{entityType: str, name: str, observations: List[str]}]
                description="List of entities to create, each with type, name, and observations.",
                required=True
            )
        ],
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["create", "entity"]})
    )
    try:
        save_custom_tool(create_entities_tool)
    except Exception as e:
        print(f"Failed to save create_entities_tool: {e}")

    # 2. create_relations
    create_relations_tool = CustomToolData(
        name="mcp_memory_create_relations",
        description="Create multiple new relations between entities in the knowledge graph.",
        script_path=common_script_path,
        parameters=[
            ToolParameter(
                name="relations",
                type="List[dict]", # [{from: str, relationType: str, to: str}]
                description="List of relations to create, each with from_entity, type, and to_entity.",
                required=True
            )
        ],
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["create", "relation"]})
    )
    try:
        save_custom_tool(create_relations_tool)
    except Exception as e:
        print(f"Failed to save create_relations_tool: {e}")

    # 3. add_observations
    add_observations_tool = CustomToolData(
        name="mcp_memory_add_observations",
        description="Add new observations to existing entities in the knowledge graph.",
        script_path=common_script_path,
        parameters=[
            ToolParameter(
                name="observations",
                type="List[dict]", # [{entityName: str, contents: List[str]}]
                description="List of observations to add, specifying entity name and content list.",
                required=True
            )
        ],
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["update", "observation"]})
    )
    try:
        save_custom_tool(add_observations_tool)
    except Exception as e:
        print(f"Failed to save add_observations_tool: {e}")

    # 4. search_nodes
    search_nodes_tool = CustomToolData(
        name="mcp_memory_search_nodes",
        description="Search for nodes (entities) in the knowledge graph based on a query.",
        script_path=common_script_path,
        parameters=[
            ToolParameter(
                name="query",
                type="str",
                description="Query string to match against entity names, types, and observations.",
                required=True
            )
        ],
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["search", "query"]})
    )
    try:
        save_custom_tool(search_nodes_tool)
    except Exception as e:
        print(f"Failed to save search_nodes_tool: {e}")

    # 5. open_nodes
    open_nodes_tool = CustomToolData(
        name="mcp_memory_open_nodes",
        description="Open/retrieve specific nodes (entities) in the knowledge graph by their names.",
        script_path=common_script_path,
        parameters=[
            ToolParameter(
                name="names",
                type="List[str]",
                description="List of entity names to retrieve.",
                required=True
            )
        ],
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["retrieve", "read"]})
    )
    try:
        save_custom_tool(open_nodes_tool)
    except Exception as e:
        print(f"Failed to save open_nodes_tool: {e}")

    # 6. read_graph
    read_graph_tool = CustomToolData(
        name="mcp_memory_read_graph",
        description="Read the entire knowledge graph.",
        script_path=common_script_path,
        parameters=[], # No input parameters typically needed, might take a dummy
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["read", "export"]})
    )
    try:
        save_custom_tool(read_graph_tool)
    except Exception as e:
        print(f"Failed to save read_graph_tool: {e}")

    # 7. delete_entities
    delete_entities_tool = CustomToolData(
        name="mcp_memory_delete_entities",
        description="Delete multiple entities and their associated relations from the knowledge graph.",
        script_path=common_script_path,
        parameters=[
            ToolParameter(
                name="entityNames",
                type="List[str]",
                description="List of entity names to delete.",
                required=True
            )
        ],
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["delete", "entity"]})
    )
    try:
        save_custom_tool(delete_entities_tool)
    except Exception as e:
        print(f"Failed to save delete_entities_tool: {e}")

    # 8. delete_relations
    delete_relations_tool = CustomToolData(
        name="mcp_memory_delete_relations",
        description="Delete multiple relations from the knowledge graph.",
        script_path=common_script_path,
        parameters=[
            ToolParameter(
                name="relations",
                type="List[dict]", # [{from: str, relationType: str, to: str}]
                description="List of relations to delete, specifying from_entity, type, and to_entity.",
                required=True
            )
        ],
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["delete", "relation"]})
    )
    try:
        save_custom_tool(delete_relations_tool)
    except Exception as e:
        print(f"Failed to save delete_relations_tool: {e}")

    # 9. delete_observations
    delete_observations_tool = CustomToolData(
        name="mcp_memory_delete_observations",
        description="Delete specific observations from entities in the knowledge graph.",
        script_path=common_script_path,
        parameters=[
            ToolParameter(
                name="deletions",
                type="List[dict]", # [{entityName: str, observations: List[str]}]
                description="List of observation deletions, specifying entity name and observation content.",
                required=True
            )
        ],
        metadata=common_metadata.model_copy(update={"tags": common_metadata.tags + ["delete", "observation"]})
    )
    try:
        save_custom_tool(delete_observations_tool)
    except Exception as e:
        print(f"Failed to save delete_observations_tool: {e}")

    # --- Saving Custom MCP Sequential Thinking Tool Definitions ---
    print("\n--- Saving Custom MCP Sequential Thinking Tool Definitions ---")

    seqthink_metadata = ToolMetadata(
        tags=["mcp", "sequential-thinking", "reasoning"],
        author="modelcontextprotocol"
    )
    seqthink_script_path = "MCP Server - SequentialThinking"  # Placeholder

    # 1. process_thought
    process_thought_tool = CustomToolData(
        name="mcp_sequentialthinking_process_thought",
        description="Process a single thought step in a sequential reasoning chain using the Sequential Thinking MCP server.",
        script_path=seqthink_script_path,
        parameters=[
            ToolParameter(
                name="thought",
                type="string",
                description="The thought or reasoning step to process.",
                required=True
            )
        ],
        metadata=seqthink_metadata.model_copy(update={"tags": seqthink_metadata.tags + ["process", "step"]})
    )
    try:
        save_custom_tool(process_thought_tool)
    except Exception as e:
        print(f"Failed to save process_thought_tool: {e}")

    # 2. generate_summary
    generate_summary_tool = CustomToolData(
        name="mcp_sequentialthinking_generate_summary",
        description="Generate a summary of the current sequential reasoning chain using the Sequential Thinking MCP server.",
        script_path=seqthink_script_path,
        parameters=[],  # No input parameters
        metadata=seqthink_metadata.model_copy(update={"tags": seqthink_metadata.tags + ["summary", "generate"]})
    )
    try:
        save_custom_tool(generate_summary_tool)
    except Exception as e:
        print(f"Failed to save generate_summary_tool: {e}")

    # 3. clear_history
    clear_history_tool = CustomToolData(
        name="mcp_sequentialthinking_clear_history",
        description="Clear the sequential reasoning history using the Sequential Thinking MCP server.",
        script_path=seqthink_script_path,
        parameters=[],  # No input parameters
        metadata=seqthink_metadata.model_copy(update={"tags": seqthink_metadata.tags + ["clear", "history"]})
    )
    try:
        save_custom_tool(clear_history_tool)
    except Exception as e:
        print(f"Failed to save clear_history_tool: {e}")

    # --- Testing the agent run endpoint ---
    print("\n--- Testing the agent run endpoint ---")
    resp = requests.post(
        "http://localhost:8000/agents/nvidia_math_agent/run",
        json={"input": "What is (7 * 8) + 12?"}
    )
    print(resp.json())