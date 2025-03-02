import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import replicate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence

# Load environment variables
load_dotenv()

# Set Replicate API token - using the exact environment variable name expected by the library
api_key = os.getenv("REPLICATE_API_KEY")
if api_key:
    os.environ["REPLICATE_API_TOKEN"] = api_key

# --- Logging Setup ---
# Configures logging to capture info and error messages for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- State Schema ---
# Defines the structure of the application state, including all necessary fields
class VaaniState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]  # List of conversation messages
    summary: str  # Summary of the conversation
    file_url: str  # URL of an attached file, if any
    web_search_enabled: bool  # Flag to enable/disable web search
    deep_research: bool  # Flag for deep research mode
    agent_name: str  # Name of the agent to route to
    extra_question: str  # Additional questions generated based on user input
    user_token: str  # User token for session management and authentication

# --- Input Validation ---
def validate_user_input(user_input: str) -> str:
    """Sanitizes user input to prevent injection attacks by allowing only safe characters."""
    return ''.join(c for c in user_input if c.isalnum() or c in (' ', '.', ',', '?', '!'))

# --- Authentication Placeholder ---
def authenticate_user(token: str) -> bool:
    """Checks user token validity. Replace with actual authentication logic (e.g., JWT)."""
    return token == "valid_token"  # Simplified example for demonstration

# --- Checkpointing with Memory Saver ---
# Sets up memory saver for state persistence instead of SQLite
memory = MemorySaver()

# --- System Prompt for Orchestrator ---
# Defines the logic for routing queries to appropriate agents
system_prompt = """
You are an orchestrator agent. Based on the user's query and context, decide which agent to route the query to.
Possible agents are:
- image_generator: if the user asks to generate an image
- rag_agent: if a file is attached (file_url is set) and the query is related to the file
- web_search_agent: if web_search_enabled is true and the query requires web search
- default: for general queries
Output only the name of the agent to route to.
"""

# --- Node Functions ---

async def extra_questions_node(state: VaaniState) -> VaaniState:
    """Generates additional questions based on the user's last message and conversation context."""
    messages = state["messages"]
    summary = state.get("summary", "")
    
    # Get the last message (which should be the user's query)
    last_message = messages[-1] if messages else None
    if last_message and last_message.type == "human":
        user_input = validate_user_input(last_message.content)
        
        # Create a prompt that includes conversation context
        context_prompt = f"Conversation summary: {summary}\n\n" if summary else ""
        
        # Add the last few messages for immediate context
        if len(messages) > 1:
            context_prompt += "Recent conversation:\n"
            for i, msg in enumerate(messages[-5:-1]):  # Skip the last message as we'll add it separately
                prefix = "User: " if msg.type == "human" else "Assistant: "
                context_prompt += f"{prefix}{msg.content}\n"
        
        prompt = f"{context_prompt}\nBased on this conversation context and the user's latest question: '{user_input}', generate 3 relevant follow-up questions."
        
        llm = ChatGroq(model_name="Llama-3.1-8b-instant")  # Lightweight model for quick responses
        try:
            response = await llm.ainvoke(prompt)
            return {"extra_question": response.content}
        except Exception as e:
            logger.error(f"Error in extra_questions_node: {e}")
            return state  # Return unchanged state on error
    return state

async def orchestrator_node(state: VaaniState) -> VaaniState:
    """Routes the query to the appropriate agent based on context and user input."""
    messages = state["messages"]
    summary = state.get("summary", "")
    file_url = state.get("file_url", "")
    web_search_enabled = state.get("web_search_enabled", False)
    deep_research = state.get("deep_research", False)
    # Build context string for the orchestrator
    context = f"Summary: {summary}\nFile attached: {'yes' if file_url else 'no'}\nWeb search enabled: {web_search_enabled}\nDeep research mode: {deep_research}"
    user_query = validate_user_input(messages[-1].content if messages else "")
    prompt = f"{system_prompt}\n\nContext: {context}\nUser query: {user_query}"
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")  # Powerful model for decision-making
    try:
        response = await llm.ainvoke(prompt)
        return {"agent_name": response.content.strip()}
    except Exception as e:
        logger.error(f"Error in orchestrator_node: {e}")
        return {"agent_name": "default_agent"}  # Fallback to default agent on error

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def image_generator_node(state: VaaniState) -> VaaniState:
    """Generates an image based on the user's query with retry logic for API failures."""
    messages = state["messages"]
    summary = state.get("summary", "")
    user_query = validate_user_input(messages[-1].content if messages else "")
    
    # Enhance the prompt with conversation context if available
    if summary or len(messages) > 1:
        # Create a context-aware prompt for image generation
        llm = ChatGroq(model_name="Llama-3.1-8b-instant")
        
        context_prompt = f"""
Based on the following conversation context, create a detailed image generation prompt that captures what the user is asking for.

Conversation summary: {summary}

Recent messages:
"""
        
        # Add the last few messages for immediate context
        for i, msg in enumerate(messages[-3:]):
            prefix = "User: " if msg.type == "human" else "Assistant: "
            context_prompt += f"{prefix}{msg.content}\n"
            
        context_prompt += "\nCreate a detailed image generation prompt based on this context:"
        
        try:
            # Get an enhanced prompt from the LLM
            response = await llm.ainvoke(context_prompt)
            enhanced_query = response.content
            logger.info(f"Enhanced image prompt: {enhanced_query[:100]}...")
            # Use the enhanced query if it's not empty
            if enhanced_query and len(enhanced_query) > 10:
                user_query = enhanced_query
        except Exception as e:
            logger.error(f"Error enhancing image prompt: {e}")
            # Continue with original query if enhancement fails
    
    try:
        # Log the API key status (without revealing the full key)
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            logger.error("REPLICATE_API_TOKEN is not set in environment variables")
            return {"messages": [AIMessage(content="Error: Replicate API token is not configured properly. Please check your environment variables.")]}
        
        # Log that we're about to make the API call
        logger.info(f"Making Replicate API call with prompt: {user_query[:50]}...")
        
        # Use the exact format from the reference implementation
        input = {"prompt": user_query}
        
        # Run Replicate API call in a thread to avoid blocking the event loop
        output = await asyncio.to_thread(replicate.run, "black-forest-labs/flux-schnell", input=input)
        
        if not output:
            logger.warning("Replicate API returned empty output")
            return {"messages": [AIMessage(content="The image generation service returned an empty response. Please try again with a different prompt.")]}
        
        # Format the output URL
        image_url = output[0] if isinstance(output, list) else output
        response = f"Generated image: {image_url}"
        logger.info("Successfully generated image")
    except Exception as e:
        import traceback
        logger.error(f"Error generating image: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Provide a more helpful error message based on the exception
        if "401" in str(e) or "Unauthenticated" in str(e):
            response = "Error: Authentication failed with Replicate. Please check your API key configuration."
        else:
            response = f"Error generating image: {str(e)}"
            
    return {"messages": [AIMessage(content=response)]}

async def rag_agent_node(state: VaaniState) -> VaaniState:
    """Processes file-based queries using a Retrieval-Augmented Generation (RAG) approach."""
    file_url = state["file_url"]
    messages = state["messages"]
    summary = state.get("summary", "")
    user_query = validate_user_input(messages[-1].content if messages else "")
    
    # Create a context-aware prompt that includes conversation history
    context_prompt = f"""
You are a helpful assistant that answers questions based on document content and conversation context.

Conversation summary: {summary}

Document: {file_url}

Please provide a detailed response to the user's query based on both the document content and the conversation context.
"""
    
    # Placeholder for actual RAG implementation
    # Steps would include:
    # 1. Parse file at file_url (e.g., PyPDF2 for PDFs)
    # 2. Extract and chunk text
    # 3. Embed chunks (e.g., with sentence-transformers)
    # 4. Store in Qdrant vector DB
    # 5. Query Qdrant with user_query
    # 6. Generate response from retrieved chunks
    
    # For now, we'll just return a placeholder response that acknowledges the conversation context
    response = f"Response based on file at {file_url} and conversation context: [Placeholder]"
    
    # In a real implementation, we would:
    # 1. Retrieve relevant chunks based on user_query
    # 2. Pass those chunks, along with conversation context, to an LLM
    # 3. Generate a response that incorporates both the document content and conversation context
    
    return {"messages": [AIMessage(content=response)]}

async def web_search_agent_node(state: VaaniState) -> VaaniState:
    """Performs a web search based on the user's query and returns the results."""
    messages = state["messages"]
    summary = state.get("summary", "")
    
    # Get the last message (user's query)
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        logger.error("Last message is not a HumanMessage")
        return {"messages": [AIMessage(content="I encountered an error processing your request.")]}
    
    user_input = last_message.content
    
    # Extract recent conversation context (last 2 exchanges)
    recent_messages = messages[-min(4, len(messages)):]
    recent_context = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in recent_messages[:-1]])
    
    try:
        # Create a search query that considers conversation context
        search_query_prompt = f"""
Based on the following conversation context and the user's latest query, generate a focused web search query.

Conversation Summary: {summary}

Recent Messages:
{recent_context}

User's Latest Query: {user_input}

Generate a concise search query that will find the most relevant information to answer the user's latest query.
"""
        
        # Use a lightweight model to generate the search query
        llm = ChatGroq(model_name="Llama-3.1-8b-instant")
        search_query_response = await llm.ainvoke([HumanMessage(content=search_query_prompt)])
        search_query = search_query_response.content.strip()
        
        logger.info(f"Generated search query: {search_query}")
        
        # Perform the web search using TavilySearchResults
        tool = TavilySearchResults(max_results=3)
        search_results = tool.invoke(search_query)  # Synchronous call to Tavily API
        
        if not search_results:
            return {"messages": [AIMessage(content="I couldn't find any relevant information on the web for your query.")]}
        
        # Format the search results for the LLM
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content available")
            url = result.get("url", "#")
            formatted_results.append(f"Result {i}:\nTitle: {title}\nContent: {content}\nURL: {url}\n")
        
        search_results_text = "\n".join(formatted_results)
        
        # Create a system message that instructs the LLM on how to format the response
        system_message = """You are a helpful AI assistant that provides information based on web search results.

IMPORTANT FORMATTING INSTRUCTIONS:
1. Analyze the search results and provide a comprehensive answer to the user's query.
2. DO NOT mention that you performed a web search unless specifically asked.
3. DO NOT include the URLs inline in your response.
4. At the end of your response, include source citations in the following format:
   [Source 1: URL1]
   [Source 2: URL2]
   etc.
5. Make sure your response is well-structured, accurate, and directly addresses the user's query.
6. Consider the conversation context when formulating your response.
"""
        
        # Create a prompt that includes the conversation context
        context_prompt = f"""
Conversation Summary: {summary}

Recent Messages:
{recent_context}

User's Latest Query: {user_input}

Based on the following web search results, please provide a comprehensive answer:

{search_results_text}
"""
        
        # Use a more powerful model for generating the final response
        response_llm = ChatGroq(model_name="Llama-3.3-70b-versatile")
        response = await response_llm.ainvoke([
            SystemMessage(content=system_message),
            HumanMessage(content=context_prompt)
        ])
        
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error(f"Error in web_search_agent_node: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"messages": [AIMessage(content="I encountered an error while searching the web. Please try again later.")]}

async def default_agent_node(state: VaaniState) -> VaaniState:
    """Handles general queries with conversation context and summary."""
    messages = state["messages"]
    summary = state.get("summary", "")
    
    # Create a system message with conversation context
    context_message = SystemMessage(content=f"""
You are a helpful AI assistant. Respond to the user's query based on the conversation context.

Conversation summary: {summary}

Remember to maintain continuity with the previous conversation and provide relevant, helpful responses.
""")
    
    # Combine context message with all conversation messages
    prompt_messages = [context_message] + messages
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")  # Versatile model for general queries
    try:
        response = await llm.ainvoke(prompt_messages)
        # Create a new AIMessage with the response content
        ai_message = AIMessage(content=response.content)
        # Return a dictionary with the messages key containing the AIMessage
        return {"messages": [ai_message]}
    except Exception as e:
        logger.error(f"Error in default_agent_node: {e}")
        return {"messages": [AIMessage(content="Error processing your request")]}

async def summarize_node(state: VaaniState) -> VaaniState:
    """Summarizes the conversation and keeps only the last 4 messages."""
    messages = state["messages"]
    summary = state.get("summary", "")
    
    # Create a prompt that asks for a comprehensive summary
    summary_prompt = f"""
You are tasked with creating a concise but comprehensive summary of the conversation so far.

Current summary: {summary}

Please update the summary to include the key points from the recent messages. 
The summary should capture:
1. Main topics discussed
2. Important information shared
3. Questions asked and answers provided
4. Any decisions or conclusions reached

Your summary will be used to provide context for future responses, so make sure it captures all relevant information.
"""
    
    llm = ChatGroq(model_name="Llama-3.1-8b-instant")  # Lightweight model for summarization
    try:
        # Add the summary prompt as a human message
        prompt_messages = messages + [HumanMessage(content=summary_prompt)]
        response = await llm.ainvoke(prompt_messages)
        
        # Remove all but the last 4 messages to manage state size
        # This ensures we keep the last 2 full exchanges (user + assistant)
        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-4]]
        
        logger.info(f"Updated summary: {response.content[:100]}...")
        return {"summary": response.content, "messages": delete_messages}
    except Exception as e:
        logger.error(f"Error in summarize_node: {e}")
        return state  # Return unchanged state on error

def summarize_check_node(state: VaaniState) -> VaaniState:
    """Passthrough node to check if summarization is needed."""
    return state

def should_summarize(state: VaaniState) -> str:
    """Decides if summarization is required based on the number of messages."""
    if len(state["messages"]) > 8:
        return "summarize"
    return "end"

# --- Workflow Setup ---
# Initializes the state graph for the Vaani.pro workflow
workflow = StateGraph(VaaniState)

# Add all nodes to the workflow
workflow.add_node("extra_questions", extra_questions_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("image_generator", image_generator_node)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("web_search_agent", web_search_agent_node)
workflow.add_node("default_agent", default_agent_node)
workflow.add_node("summarize_check", summarize_check_node)
workflow.add_node("summarize", summarize_node)

# Define workflow edges
workflow.set_entry_point("extra_questions")  # Start with generating extra questions
workflow.add_edge("extra_questions", "orchestrator")  # Route to orchestrator next

def route_to_agent(state: VaaniState) -> str:
    """Routes to the selected agent, falling back to 'default_agent' if invalid."""
    agent_name = state["agent_name"]
    valid_agents = {"image_generator", "rag_agent", "web_search_agent", "default_agent"}
    return agent_name if agent_name in valid_agents else "default_agent"

# Conditional routing from orchestrator to specific agents
workflow.add_conditional_edges("orchestrator", route_to_agent, {
    "image_generator": "image_generator",
    "rag_agent": "rag_agent",
    "web_search_agent": "web_search_agent",
    "default_agent": "default_agent"
})

# Connect all agents to summarize_check node
for agent in ["image_generator", "rag_agent", "web_search_agent", "default_agent"]:
    workflow.add_edge(agent, "summarize_check")

# Conditional edge to either summarize or end the workflow
workflow.add_conditional_edges("summarize_check", should_summarize, {
    "summarize": "summarize",
    "end": END
})
workflow.add_edge("summarize", END)  # End workflow after summarization

# Compile the Graph with checkpointing
graph = workflow.compile(checkpointer=memory)

# --- Example Usage ---
async def run_example():
    """Runs an example query through the workflow."""
    config = {"configurable": {"thread_id": "1"}}  # Thread ID for state persistence
    state_update = {
        "messages": [HumanMessage(content="Generate an image of a dog")],
        "summary": "",
        "file_url": "",
        "web_search_enabled": False,
        "deep_research": False,
        "agent_name": "",
        "extra_question": "",
        "user_token": "valid_token"  # Example token for authentication
    }
    # Authenticate user before processing
    if not authenticate_user(state_update["user_token"]):
        logger.error("Authentication failed")
        return
    # Run the workflow and print the final message
    output = await graph.ainvoke(state_update, config)
    for m in output["messages"][-1:]:
        m.pretty_print()

if __name__ == "__main__":
    asyncio.run(run_example())

# Export the graph and VaaniState for use in other modules
__all__ = ["graph", "VaaniState"]