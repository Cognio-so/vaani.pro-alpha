# Cognio Agent

An intelligent conversational agent framework with multi-agent orchestration capabilities.

## Overview

Cognio Agent is an advanced AI assistant platform that dynamically routes queries to specialized agents for optimal response generation. Built with a state-of-the-art architecture using LangGraph and LangChain, it offers robust conversation management with state persistence.

## Built By

Ashutosh Upadhyay  
Cognio Labs

## Features

- **Multi-Agent Orchestration**: Automatically routes queries to specialized agents
- **Specialized Capabilities**:
  - Image Generation
  - Document Analysis (RAG)
  - Web Search
  - General Conversation
- **Advanced Conversation Management**:
  - Automatic summarization
  - Context retention
  - Follow-up question generation
- **Robust Infrastructure**:
  - SQLite-based state persistence
  - Error handling with retry logic
  - Secure input validation

## Technical Architecture

Cognio Agent uses a modular architecture powered by:
- **LangGraph**: For workflow orchestration and state management
- **LangChain**: For LLM interaction and tool integration
- **Groq**: High-performance LLM inference
- **Replicate**: For image generation capabilities
- **Tavily**: For web search functionality

## Agent Types

1. **Orchestrator Agent**: Routes queries to appropriate specialized agents
2. **Image Generator**: Creates images based on text descriptions
3. **RAG Agent**: Analyzes documents and answers questions about them
4. **Web Search Agent**: Retrieves and synthesizes information from the web
5. **Default Agent**: Handles general conversation

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/cogniolabs/cognio-agent.git
cd cognio-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- GROQ_API_KEY
- REPLICATE_API_KEY
- TAVILY_API_KEY

## Usage

```python
import asyncio
from agent import graph, VaaniState
from langchain_core.messages import HumanMessage

async def main():
    # Initialize state
    state = {
        "messages": [HumanMessage(content="Hello, can you help me with a research question?")],
        "summary": "",
        "file_url": "",
        "web_search_enabled": True,
        "deep_research": False,
        "agent_name": "",
        "extra_question": "",
        "user_token": "your_auth_token"
    }
    
    # Run the agent
    config = {"configurable": {"thread_id": "user_session_123"}}
    result = await graph.ainvoke(state, config)
    
    # Print the response
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Configuration

The agent behavior can be customized through the following parameters:
- `web_search_enabled`: Enable/disable web search capabilities
- `deep_research`: Enable more thorough research mode
- `file_url`: URL to a document for RAG processing

## License

Copyright © 2023 Cognio Labs. All rights reserved. 