# 🦜🔗 LangGraph — Complete Notes: Generative AI & Agentic AI
### From Absolute Beginner → Production-Ready Systems

> **Version:** LangGraph `0.2.x / 0.3.x` | Python 3.10+  
> **Last Updated:** 2025  
> **Audience:** Beginners with basic Python knowledge

---

## 📚 Table of Contents

1. [What is LangGraph?](#1-what-is-langgraph)
2. [Why LangGraph over LangChain Alone?](#2-why-langgraph-over-langchain-alone)
3. [Core Mental Model](#3-core-mental-model)
4. [Installation & Setup](#4-installation--setup)
5. [Key Building Blocks](#5-key-building-blocks)
6. [Your First Graph — Hello World](#6-your-first-graph--hello-world)
7. [State Management](#7-state-management)
8. [Nodes in Depth](#8-nodes-in-depth)
9. [Edges in Depth](#9-edges-in-depth)
10. [Conditional Routing](#10-conditional-routing)
11. [Tool Calling & Function Use](#11-tool-calling--function-use)
12. [Memory & Checkpointing (Persistence)](#12-memory--checkpointing-persistence)
13. [Human-in-the-Loop (HITL)](#13-human-in-the-loop-hitl)
14. [Multi-Agent Systems](#14-multi-agent-systems)
15. [Streaming](#15-streaming)
16. [Common Graph Patterns](#16-common-graph-patterns)
17. [Error Handling & Retries](#17-error-handling--retries)
18. [Debugging & Testing](#18-debugging--testing)
19. [LangGraph Platform (Cloud / Studio)](#19-langgraph-platform-cloud--studio)
20. [Production Deployment](#20-production-deployment)
21. [Best Practices](#21-best-practices)
22. [Glossary](#22-glossary)
23. [Resources & References](#23-resources--references)

---

## 1. What is LangGraph?

**LangGraph** is an open-source framework by LangChain Inc. for building **stateful, multi-step, multi-agent AI applications** using a **graph-based execution model**.

Think of it this way:

```
Traditional LLM Call:   Input → LLM → Output  (one shot)

LangGraph:              Input → Node1 → Node2 → Decision → Node3 → Output
                                    ↑__________________________|  (loops possible)
```

### What problems does it solve?

| Problem | LangGraph Solution |
|---|---|
| LLMs forget context mid-task | Persistent shared **State** |
| Can't loop / retry intelligently | **Cycles** in the graph |
| Hard to branch logic | **Conditional edges** |
| Multi-agent coordination is complex | **Supervisor / Subgraph** patterns |
| Debugging black-box chains | **Visual Studio + streaming** |
| No human approval mid-task | **Human-in-the-Loop** breakpoints |

---

## 2. Why LangGraph over LangChain Alone?

| Feature | LangChain (LCEL) | LangGraph |
|---|---|---|
| Execution model | Linear / sequential DAG | Graph with **cycles** |
| State sharing | Limited | First-class `TypedDict` state |
| Loops & retries | Difficult | Native support |
| Human approval | Not built-in | Built-in breakpoints |
| Multi-agent | Manual wiring | Supervisor / handoff patterns |
| Persistence | Manual | Checkpointer built-in |
| Streaming | Token-level | Node + token level |

> **Rule of thumb:** Use LangChain for simple chains. Use LangGraph when your app needs loops, branches, memory, or multiple agents.

---

## 3. Core Mental Model

LangGraph is based on **state machines** and **directed graphs**.

```
┌─────────────────────────────────────────────┐
│                  GRAPH                       │
│                                             │
│   START ──→ [Node A] ──→ [Node B] ──→ END  │
│                  ↑           │              │
│                  └───────────┘              │
│                   (cycle / loop)            │
│                                             │
│  State flows through every node             │
└─────────────────────────────────────────────┘
```

### Three pillars:

```
1. STATE      →  Shared memory / data object passed between nodes
2. NODES      →  Python functions that READ and WRITE to state
3. EDGES      →  Connections between nodes (fixed or conditional)
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager
- An LLM API key (OpenAI, Anthropic, Google, etc.)

### Install

```bash
# Core LangGraph
pip install langgraph

# LangChain + LLM integrations
pip install langchain langchain-openai langchain-anthropic

# Optional: LangSmith for tracing/debugging
pip install langsmith

# Optional: LangGraph CLI for platform deployment
pip install langgraph-cli
```

### Environment variables

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=ls__...        # LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-langgraph-app
```

```python
# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

### Verify installation

```python
import langgraph
print(langgraph.__version__)  # e.g., 0.2.56
```

---

## 5. Key Building Blocks

### 5.1 The Big Picture

```
┌──────────────────────────────────────────────────────┐
│  LangGraph Application                               │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │
│  │  State  │ ←→ │  Nodes  │ ←→ │  Edges          │  │
│  │(TypedDict│    │(functions│   │(Normal +        │  │
│  │ or      │    │ or async │   │ Conditional)    │  │
│  │ BaseModel│   │ callables│   │                 │  │
│  └─────────┘    └─────────┘    └─────────────────┘  │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │
│  │Checkpointer  │  Tools  │    │ Human-in-Loop   │  │
│  │(memory) │    │(functions│   │(breakpoints)    │  │
│  └─────────┘    └─────────┘    └─────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 5.2 Quick Reference Table

| Concept | What it is | Python type |
|---|---|---|
| `State` | Shared data between nodes | `TypedDict` or `BaseModel` |
| `Node` | A function that processes state | `Callable[[State], dict]` |
| `Edge` | Connection between nodes | `graph.add_edge(a, b)` |
| `Conditional Edge` | Branch based on state | `graph.add_conditional_edges(...)` |
| `START` | Entry point constant | `langgraph.graph.START` |
| `END` | Exit point constant | `langgraph.graph.END` |
| `StateGraph` | The graph object | `StateGraph(MyState)` |
| `CompiledGraph` | Runnable graph | `graph.compile()` |
| `Checkpointer` | Persistence layer | `MemorySaver`, `SqliteSaver` |

---

## 6. Your First Graph — Hello World

```python
# hello_langgraph.py

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ─────────────────────────────────────────
# STEP 1: Define your State
# ─────────────────────────────────────────
class MyState(TypedDict):
    message: str
    response: str

# ─────────────────────────────────────────
# STEP 2: Define Nodes (plain functions)
# ─────────────────────────────────────────
def greet_node(state: MyState) -> dict:
    """Reads state, returns partial update."""
    name = state["message"]
    return {"response": f"Hello, {name}! Welcome to LangGraph."}

# ─────────────────────────────────────────
# STEP 3: Build the Graph
# ─────────────────────────────────────────
builder = StateGraph(MyState)

# Add nodes
builder.add_node("greet", greet_node)

# Add edges
builder.add_edge(START, "greet")
builder.add_edge("greet", END)

# Compile
graph = builder.compile()

# ─────────────────────────────────────────
# STEP 4: Run the Graph
# ─────────────────────────────────────────
result = graph.invoke({"message": "Alice", "response": ""})
print(result)
# {'message': 'Alice', 'response': 'Hello, Alice! Welcome to LangGraph.'}
```

### Visualize the graph (Jupyter)

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

---

## 7. State Management

State is the **heart** of LangGraph. It's a shared data object passed between every node.

### 7.1 TypedDict (Simple)

```python
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]        # chat history
    current_step: str          # which step we're on
    tool_results: List[dict]   # results from tools
    final_answer: str          # output
    error: str                 # any error
```

### 7.2 With Annotated (Reducers) ⭐ Important!

By default, each node **replaces** the state key it returns.  
Use `Annotated` + a **reducer function** to **merge/append** instead.

```python
from typing import Annotated
import operator

class AgentState(TypedDict):
    # operator.add → appends lists instead of replacing
    messages: Annotated[list, operator.add]
    # Default → replaces the value
    status: str
```

```python
# Without reducer (replace):
# Before: messages = ["hi"]
# Node returns: {"messages": ["how are you"]}
# After: messages = ["how are you"]  ← REPLACED

# With Annotated[list, operator.add] (append):
# Before: messages = ["hi"]
# Node returns: {"messages": ["how are you"]}
# After: messages = ["hi", "how are you"]  ← APPENDED
```

### 7.3 LangChain Messages State (Most Common for Chat)

```python
from langgraph.graph import MessagesState
# OR manually:
from langchain_core.messages import BaseMessage
from typing import Annotated
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages is a smart reducer that handles:
    # - Appending new messages
    # - Updating existing messages by ID
```

### 7.4 Pydantic State (Validation)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class StrictState(BaseModel):
    query: str
    steps_taken: int = 0
    results: List[str] = Field(default_factory=list)
    is_complete: bool = False
```

### 7.5 State Best Practices

```python
# ✅ DO: Return only the keys you're updating
def my_node(state: AgentState) -> dict:
    return {"messages": [new_message]}  # partial update

# ❌ DON'T: Return the whole state (causes bugs with reducers)
def bad_node(state: AgentState) -> AgentState:
    state["messages"].append(new_message)
    return state  # don't do this

# ✅ DO: Access state as a dict
def good_node(state: AgentState) -> dict:
    current_messages = state["messages"]
    user_query = state.get("query", "")
    return {"status": "processed"}
```

---

## 8. Nodes in Depth

A **node** is any Python callable that:
- Takes `state` as input
- Returns a `dict` with partial state updates

### 8.1 Basic Node

```python
def simple_node(state: MyState) -> dict:
    return {"key": "value"}
```

### 8.2 Node with LLM Call

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_node(state: AgentState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)        # Call the LLM
    return {"messages": [response]}        # Return AIMessage
```

### 8.3 Async Node

```python
import asyncio

async def async_llm_node(state: AgentState) -> dict:
    messages = state["messages"]
    response = await llm.ainvoke(messages)   # Async call
    return {"messages": [response]}

# Run async graph
result = await graph.ainvoke({"messages": [...]})
```

### 8.4 Node with External API

```python
import httpx

def fetch_data_node(state: AgentState) -> dict:
    query = state["query"]
    
    with httpx.Client() as client:
        resp = client.get(f"https://api.example.com/search?q={query}")
        data = resp.json()
    
    return {"results": data["items"], "status": "fetched"}
```

### 8.5 Adding Nodes

```python
builder = StateGraph(AgentState)

# Method 1: Direct function reference
builder.add_node("my_node", my_function)

# Method 2: Lambda (for quick transforms)
builder.add_node("format", lambda state: {"output": state["raw"].upper()})

# Method 3: Class with __call__
class MyNode:
    def __call__(self, state):
        return {"result": "done"}

builder.add_node("custom", MyNode())
```

---

## 9. Edges in Depth

Edges connect nodes and control execution flow.

### 9.1 Normal Edges (Always execute)

```python
builder.add_edge("node_a", "node_b")   # a → b, always
builder.add_edge(START, "first_node")  # Entry point
builder.add_edge("last_node", END)     # Exit point
```

### 9.2 Multiple Outgoing Edges (Fan-out / Parallel)

```python
# Both node_b and node_c run after node_a (parallel branches)
builder.add_edge("node_a", "node_b")
builder.add_edge("node_a", "node_c")
# Then node_d waits for both
builder.add_edge("node_b", "node_d")
builder.add_edge("node_c", "node_d")
```

### 9.3 Conditional Edges (Routing) ⭐

```python
def route_function(state: AgentState) -> str:
    """Returns the NAME of the next node to visit."""
    if state["needs_tool"]:
        return "tool_node"
    else:
        return "answer_node"

builder.add_conditional_edges(
    "decision_node",          # Source node
    route_function,           # Router function
    {
        "tool_node": "tool_node",     # Optional: map return values
        "answer_node": "answer_node"  # to node names
    }
)
```

### 9.4 Entry Point

```python
# Method 1: Using START constant
builder.add_edge(START, "first_node")

# Method 2: set_entry_point (older API)
builder.set_entry_point("first_node")
```

### 9.5 Finish Point

```python
# Method 1: Using END constant
builder.add_edge("last_node", END)

# Method 2: set_finish_point
builder.set_finish_point("last_node")

# Method 3: Conditional → END
def router(state):
    if state["done"]:
        return END
    return "continue_node"
```

---

## 10. Conditional Routing

Conditional routing is how LangGraph makes decisions.

### 10.1 Simple Router

```python
from langgraph.graph import END

def should_continue(state: AgentState) -> str:
    """Decides which node to go to next."""
    last_message = state["messages"][-1]
    
    # If LLM called a tool, go to tool executor
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, we're done
    return END

builder.add_conditional_edges("agent", should_continue)
```

### 10.2 Multi-branch Router

```python
def classify_intent(state: AgentState) -> str:
    query = state["query"].lower()
    
    if "weather" in query:
        return "weather_agent"
    elif "calculate" in query or "math" in query:
        return "math_agent"
    elif "search" in query:
        return "search_agent"
    else:
        return "general_agent"

builder.add_conditional_edges(
    "classifier",
    classify_intent,
    {
        "weather_agent": "weather_agent",
        "math_agent": "math_agent",
        "search_agent": "search_agent",
        "general_agent": "general_agent",
    }
)
```

### 10.3 Loop Until Done Pattern

```python
class IterativeState(TypedDict):
    messages: Annotated[list, operator.add]
    iterations: int
    max_iterations: int
    is_done: bool

def check_done(state: IterativeState) -> str:
    if state["is_done"]:
        return END
    if state["iterations"] >= state["max_iterations"]:
        return END        # Safety: prevent infinite loops
    return "agent"        # Loop back

builder.add_conditional_edges("agent", check_done)
```

---

## 11. Tool Calling & Function Use

Tools let the LLM interact with the real world (search, APIs, code execution, databases).

### 11.1 Define Tools

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information. Use when you need current data."""
    # In practice, integrate with Tavily, SerpAPI, etc.
    return f"Search results for '{query}': [result1, result2]"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: a valid Python math expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    # Call a real weather API here
    return {"city": city, "temp": "22°C", "condition": "Sunny"}

tools = [search_web, calculator, get_weather]
```

### 11.2 Bind Tools to LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)   # LLM now knows about tools
```

### 11.3 ToolNode — Automatic Tool Execution

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)   # Handles tool execution automatically

builder.add_node("tools", tool_node)
```

### 11.4 Full ReAct Agent Pattern

```python
from typing import Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# --- Tools ---
@tool
def search(query: str) -> str:
    """Search for information online."""
    return f"Found: [results for {query}]"

tools = [search]

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# --- Nodes ---
def agent_node(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# --- Graph ---
builder = StateGraph(MessagesState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

# tools_condition checks if last message has tool_calls
builder.add_conditional_edges("agent", tools_condition)

builder.add_edge("tools", "agent")   # After tools → back to agent

graph = builder.compile()

# --- Run ---
from langchain_core.messages import HumanMessage

result = graph.invoke({
    "messages": [HumanMessage(content="Search for LangGraph tutorials")]
})
print(result["messages"][-1].content)
```

### 11.5 Using create_react_agent (Shortcut) ⭐

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Creates a full ReAct agent in one line
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are a helpful assistant."  # System prompt
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 2 + 2?"}]
})
```

---

## 12. Memory & Checkpointing (Persistence)

Checkpointing lets your graph **remember** between runs (conversation history, state).

### 12.1 Memory Types

```
┌─────────────────────────────────────────────────┐
│               Memory in LangGraph               │
├─────────────────┬───────────────────────────────┤
│ In-Scope Memory │ State within one graph run    │
│                 │ (automatic)                   │
├─────────────────┼───────────────────────────────┤
│ Cross-Run Memory│ State saved between runs      │
│                 │ (requires Checkpointer)        │
└─────────────────┴───────────────────────────────┘
```

### 12.2 MemorySaver (In-Memory, Dev/Testing)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)

# Thread ID groups related conversations
config = {"configurable": {"thread_id": "user-alice-session-1"}}

# First run
result1 = graph.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is Alice"}]},
    config=config
)

# Second run — graph remembers previous messages!
result2 = graph.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config=config   # same thread_id
)
print(result2["messages"][-1].content)  # "Your name is Alice"
```

### 12.3 SqliteSaver (File Persistence)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Saves to a local SQLite file
with SqliteSaver.from_conn_string("./checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    
    result = graph.invoke(
        {"messages": [...]},
        config={"configurable": {"thread_id": "session-1"}}
    )
```

### 12.4 PostgresSaver (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

conn_string = "postgresql://user:password@localhost:5432/mydb"

with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    checkpointer.setup()   # Create tables if not exist
    graph = builder.compile(checkpointer=checkpointer)
```

### 12.5 Inspecting State History

```python
config = {"configurable": {"thread_id": "user-123"}}

# Get current state
current_state = graph.get_state(config)
print(current_state.values)        # State dict
print(current_state.next)          # Next nodes to run
print(current_state.metadata)      # Step, run_id, etc.

# Get full history
history = list(graph.get_state_history(config))
for step in history:
    print(f"Step {step.metadata['step']}: {step.values}")

# Time travel: run from a specific checkpoint
past_state = history[2]
graph.invoke(None, config={"configurable": {"thread_id": "user-123",
                                            "checkpoint_id": past_state.config["configurable"]["checkpoint_id"]}})
```

### 12.6 Thread vs Session Management

```python
# Each thread_id = isolated conversation
config_alice = {"configurable": {"thread_id": "alice"}}
config_bob   = {"configurable": {"thread_id": "bob"}}

# These are completely isolated — Alice's history ≠ Bob's history
graph.invoke({"messages": [...]}, config=config_alice)
graph.invoke({"messages": [...]}, config=config_bob)
```

---

## 13. Human-in-the-Loop (HITL)

Let humans review, approve, or modify AI decisions mid-execution.

### 13.1 Interrupt (Pause at a Node)

```python
# Compile with interrupt BEFORE a sensitive node
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send_email_node"]   # Pause before this runs
)

config = {"configurable": {"thread_id": "task-1"}}

# Graph runs until it hits "send_email_node", then pauses
result = graph.invoke({"messages": [...]}, config=config)
# result will be None or partial — graph is paused

# Inspect what's about to happen
state = graph.get_state(config)
print("Email content:", state.values["draft_email"])
print("Next nodes:", state.next)  # → ["send_email_node"]

# Human reviews and approves — resume by invoking with None
final_result = graph.invoke(None, config=config)
```

### 13.2 Interrupt After (Pause after, before next)

```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["draft_node"]   # Pause after draft_node, before next
)
```

### 13.3 Human Edits State Before Resuming

```python
# Pause before sending
state = graph.get_state(config)

# Human sees the draft and wants to modify it
print("Current draft:", state.values["draft_email"])

# Update state with human correction
graph.update_state(
    config,
    {"draft_email": "Dear Alice, [CORRECTED VERSION]..."}
)

# Resume with updated state
graph.invoke(None, config=config)
```

### 13.4 Human Node Pattern (Explicit Approval)

```python
from langgraph.types import interrupt

def human_approval_node(state: AgentState) -> dict:
    """Pause here and wait for human input."""
    # interrupt() pauses the graph and surfaces info to the user
    human_response = interrupt({
        "question": "Should I proceed with this action?",
        "proposed_action": state["planned_action"],
        "data": state["action_data"]
    })
    
    return {"human_approved": human_response == "yes",
            "human_feedback": human_response}

builder.add_node("human_approval", human_approval_node)

# Resume with human input
from langgraph.types import Command
graph.invoke(Command(resume="yes"), config=config)
```

---

## 14. Multi-Agent Systems

Multiple specialized agents working together is where LangGraph truly shines.

### 14.1 Patterns Overview

```
┌─────────────────────────────────────────────────────┐
│            Multi-Agent Architectures                │
├─────────────────────────────────────────────────────┤
│  1. SEQUENTIAL   A → B → C (pipeline)               │
│  2. SUPERVISOR   Orchestrator → picks agent         │
│  3. HIERARCHICAL Supervisor of Supervisors          │
│  4. NETWORK      Any agent → any agent              │
│  5. SWARM        Agents hand off dynamically        │
└─────────────────────────────────────────────────────┘
```

### 14.2 Supervisor Pattern ⭐ (Most Common)

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal

# Specialized agents
def research_agent(state):
    """Searches and retrieves information."""
    # ... implementation
    return {"messages": [AIMessage(content="Research results: ...")]}

def writer_agent(state):
    """Writes and formats content."""
    # ... implementation
    return {"messages": [AIMessage(content="Draft: ...")]}

def reviewer_agent(state):
    """Reviews and QA's content."""
    # ... implementation
    return {"messages": [AIMessage(content="Review: ...")]}

# Supervisor decides who goes next
supervisor_llm = ChatOpenAI(model="gpt-4o")

def supervisor(state: MessagesState) -> dict:
    system = """You are a supervisor managing: researcher, writer, reviewer.
    Based on the conversation, decide who should act next.
    Respond with ONLY one of: researcher, writer, reviewer, FINISH"""
    
    response = supervisor_llm.invoke([
        {"role": "system", "content": system},
        *state["messages"]
    ])
    return {"messages": [response], "next_agent": response.content.strip()}

def route_to_agent(state) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent

# Build graph
class SupervisorState(MessagesState):
    next_agent: str

builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", research_agent)
builder.add_node("writer", writer_agent)
builder.add_node("reviewer", reviewer_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_to_agent)
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("reviewer", "supervisor")

graph = builder.compile()
```

### 14.3 Subgraphs (Nesting Graphs)

```python
# --- Inner subgraph ---
inner_builder = StateGraph(InnerState)
inner_builder.add_node("step1", step1_fn)
inner_builder.add_node("step2", step2_fn)
inner_builder.add_edge(START, "step1")
inner_builder.add_edge("step1", "step2")
inner_builder.add_edge("step2", END)
inner_graph = inner_builder.compile()

# --- Outer graph uses inner graph as a node ---
outer_builder = StateGraph(OuterState)
outer_builder.add_node("pre_process", pre_process_fn)
outer_builder.add_node("inner_workflow", inner_graph)   # ← subgraph as node!
outer_builder.add_node("post_process", post_process_fn)

outer_builder.add_edge(START, "pre_process")
outer_builder.add_edge("pre_process", "inner_workflow")
outer_builder.add_edge("inner_workflow", "post_process")
outer_builder.add_edge("post_process", END)

outer_graph = outer_builder.compile()
```

### 14.4 Handoff Pattern (Agent-to-Agent)

```python
from langgraph.types import Command

def agent_a(state) -> Command:
    # Decide to hand off to agent_b
    if needs_specialist:
        return Command(
            goto="agent_b",             # Transfer control
            update={"messages": [...]}  # Pass context
        )
    return {"messages": [...]}         # Continue normally

def agent_b(state):
    # Receives control from agent_a
    return {"messages": [...]}

builder.add_node("agent_a", agent_a)
builder.add_node("agent_b", agent_b)
```

### 14.5 Parallel Execution (Fan-Out / Fan-In)

```python
import operator
from typing import Annotated

class ParallelState(TypedDict):
    query: str
    research_result: str
    calculation_result: str
    final_answer: str

def research_agent(state):
    return {"research_result": f"Research on: {state['query']}"}

def calculator_agent(state):
    return {"calculation_result": f"Calculation for: {state['query']}"}

def synthesizer(state):
    combined = f"{state['research_result']} + {state['calculation_result']}"
    return {"final_answer": combined}

builder = StateGraph(ParallelState)
builder.add_node("research", research_agent)
builder.add_node("calculator", calculator_agent)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "research")
builder.add_edge(START, "calculator")       # Both run in parallel!
builder.add_edge("research", "synthesizer")  # Wait for both
builder.add_edge("calculator", "synthesizer")

graph = builder.compile()
```

---

## 15. Streaming

Stream results token by token or node by node for responsive UX.

### 15.1 Stream Modes

```python
# mode="values"   → full state after each node
# mode="updates"  → only the delta (what changed)
# mode="messages" → token-by-token streaming (for LLM nodes)
# mode="debug"    → everything (verbose)
```

### 15.2 Stream Updates (Node-by-node)

```python
config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "Tell me a joke"}]},
    config=config,
    stream_mode="updates"
):
    for node_name, node_output in chunk.items():
        print(f"\n[Node: {node_name}]")
        print(node_output)
```

### 15.3 Stream Tokens (Real-time)

```python
async def stream_tokens():
    async for event in graph.astream_events(
        {"messages": [{"role": "user", "content": "Hello!"}]},
        version="v2"
    ):
        kind = event["event"]
        
        # Stream LLM tokens
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        
        # Node started
        elif kind == "on_chain_start":
            print(f"\n>> Node starting: {event['name']}")
        
        # Node finished
        elif kind == "on_chain_end":
            print(f"\n<< Node done: {event['name']}")

import asyncio
asyncio.run(stream_tokens())
```

### 15.4 Stream in FastAPI (Production)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/chat")
async def chat_stream(request: dict):
    async def generate():
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": request["message"]}]},
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 16. Common Graph Patterns

### 16.1 Pattern: Reflection / Self-Critique

```python
# Agent generates → Critic reviews → Revise if needed → Done

class ReflectionState(TypedDict):
    messages: Annotated[list, operator.add]
    draft: str
    critique: str
    revision_count: int

def generator(state):
    response = llm.invoke(state["messages"])
    return {"draft": response.content, "messages": [response]}

def critic(state):
    critique_prompt = f"Critique this draft:\n{state['draft']}\n\nBe specific."
    critique = llm.invoke([HumanMessage(content=critique_prompt)])
    return {"critique": critique.content, "revision_count": state["revision_count"] + 1}

def should_revise(state) -> str:
    if state["revision_count"] >= 3:
        return END
    if "good" in state["critique"].lower() or "excellent" in state["critique"].lower():
        return END
    return "generator"   # Revise

builder.add_node("generator", generator)
builder.add_node("critic", critic)
builder.add_edge(START, "generator")
builder.add_edge("generator", "critic")
builder.add_conditional_edges("critic", should_revise)
```

### 16.2 Pattern: Plan-and-Execute

```python
# Planner creates steps → Executor runs each → Aggregator collects results

def planner(state):
    plan = llm.invoke(f"Create a step-by-step plan for: {state['goal']}")
    steps = parse_steps(plan.content)
    return {"plan": steps, "current_step": 0}

def executor(state):
    step = state["plan"][state["current_step"]]
    result = run_step(step)
    return {"results": [result], "current_step": state["current_step"] + 1}

def check_plan_done(state) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "aggregator"
    return "executor"

def aggregator(state):
    final = "\n".join(state["results"])
    return {"final_answer": final}
```

### 16.3 Pattern: RAG (Retrieval Augmented Generation)

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

class RAGState(TypedDict):
    question: str
    context: str
    answer: str

def retrieve_node(state: RAGState) -> dict:
    docs = vectorstore.similarity_search(state["question"], k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generate_node(state: RAGState) -> dict:
    prompt = f"""Answer using the context below:
    
Context: {state['context']}

Question: {state['question']}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
```

### 16.4 Pattern: Map-Reduce

```python
# Split work → Process in parallel → Combine results

from langgraph.constants import Send

class MapReduceState(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], operator.add]
    final_summary: str

def split_documents(state: MapReduceState):
    """Fan-out: Send each doc to summarizer."""
    return [
        Send("summarize", {"document": doc})
        for doc in state["documents"]
    ]

def summarize(state):
    doc = state["document"]
    summary = llm.invoke(f"Summarize: {doc}").content
    return {"summaries": [summary]}

def combine(state: MapReduceState):
    all_summaries = "\n".join(state["summaries"])
    final = llm.invoke(f"Combine these summaries:\n{all_summaries}").content
    return {"final_summary": final}

builder = StateGraph(MapReduceState)
builder.add_node("summarize", summarize)
builder.add_node("combine", combine)

builder.add_conditional_edges(START, split_documents, ["summarize"])
builder.add_edge("summarize", "combine")
builder.add_edge("combine", END)
```

---

## 17. Error Handling & Retries

### 17.1 Try-Except in Nodes

```python
def safe_api_node(state: AgentState) -> dict:
    try:
        result = call_external_api(state["query"])
        return {"api_result": result, "error": None}
    except Exception as e:
        return {"api_result": None, "error": str(e)}

def handle_error_routing(state: AgentState) -> str:
    if state.get("error"):
        return "fallback_node"
    return "process_node"
```

### 17.2 Retry Logic

```python
import time
from functools import wraps

def with_retry(max_retries=3, delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            for attempt in range(max_retries):
                try:
                    return func(state)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
        return wrapper
    return decorator

@with_retry(max_retries=3)
def flaky_node(state: AgentState) -> dict:
    return {"result": call_unreliable_api()}
```

### 17.3 Fallback Node Pattern

```python
class RobustState(TypedDict):
    messages: Annotated[list, operator.add]
    error_count: int
    max_errors: int

def primary_llm_node(state):
    try:
        resp = expensive_llm.invoke(state["messages"])
        return {"messages": [resp], "error_count": 0}
    except Exception as e:
        return {"error_count": state["error_count"] + 1}

def fallback_llm_node(state):
    resp = cheap_llm.invoke(state["messages"])
    return {"messages": [resp]}

def route_after_primary(state) -> str:
    if state["error_count"] > 0:
        return "fallback"
    return END

builder.add_conditional_edges("primary", route_after_primary)
```

---

## 18. Debugging & Testing

### 18.1 LangSmith Tracing (Recommended)

```python
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=my-project-name

# Every graph.invoke() will be traced automatically
# View at: https://smith.langchain.com
```

### 18.2 Print Streaming Debug

```python
for step in graph.stream({"messages": [...]}, stream_mode="updates"):
    print("=" * 40)
    for node_name, output in step.items():
        print(f"NODE: {node_name}")
        print(f"OUTPUT: {output}")
```

### 18.3 Visualize Graph Structure

```python
# In Jupyter/Colab
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# Print as ASCII
print(graph.get_graph().draw_ascii())

# Print as Mermaid code
print(graph.get_graph().draw_mermaid())
```

### 18.4 Unit Testing Nodes

```python
import pytest

def test_greet_node():
    state = {"message": "Alice", "response": ""}
    result = greet_node(state)
    assert result["response"] == "Hello, Alice! Welcome to LangGraph."

def test_routing():
    state = {"messages": [AIMessage(content="hello", tool_calls=[])]}
    assert should_continue(state) == END
    
    state_with_tools = {"messages": [AIMessage(content="", tool_calls=[{"name": "search"}])]}
    assert should_continue(state_with_tools) == "tools"

def test_full_graph():
    result = graph.invoke({"messages": [HumanMessage(content="Hi")]})
    assert "messages" in result
    assert len(result["messages"]) > 0
```

### 18.5 State Inspection Utilities

```python
def debug_state(state: dict):
    """Pretty-print current state."""
    print("\n🔍 Current State:")
    for key, value in state.items():
        if isinstance(value, list):
            print(f"  {key}: [{len(value)} items]")
            for i, item in enumerate(value[-2:]):  # Show last 2
                print(f"    [{i}]: {str(item)[:100]}...")
        else:
            print(f"  {key}: {str(value)[:100]}")
    print()
```

---

## 19. LangGraph Platform (Cloud / Studio)

### 19.1 LangGraph Studio (Local UI)

Visual debugger for your graphs. Run and inspect graphs with a GUI.

```bash
# Install
pip install "langgraph-cli[inmem]"

# Run Studio (in your project directory)
langgraph dev
# Opens at http://localhost:8123/studio
```

### 19.2 Project Structure for Platform

```
my-langgraph-app/
├── langgraph.json          ← Config file (required)
├── my_agent/
│   ├── __init__.py
│   ├── graph.py            ← Main graph definition
│   ├── state.py            ← State definitions
│   ├── nodes.py            ← Node functions
│   └── tools.py            ← Tool definitions
├── requirements.txt
└── .env
```

### 19.3 langgraph.json

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./my_agent/graph.py:graph"
  },
  "env": ".env"
}
```

### 19.4 Deploy to LangGraph Cloud

```bash
# Login
langgraph auth

# Deploy
langgraph deploy

# Check status
langgraph status
```

### 19.5 LangGraph SDK (Call Deployed Graph)

```python
from langgraph_sdk import get_client

client = get_client(url="https://your-deployment.langchain.com")

# List available assistants
assistants = await client.assistants.search()

# Create a thread
thread = await client.threads.create()

# Run
async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input={"messages": [{"role": "user", "content": "Hello!"}]},
    stream_mode="messages"
):
    print(chunk)
```

---

## 20. Production Deployment

### 20.1 Self-Hosted with FastAPI

```python
# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.checkpoint.postgres import PostgresSaver
import json, asyncio

# Production checkpointer
checkpointer = PostgresSaver.from_conn_string(os.getenv("DATABASE_URL"))
checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)

app = FastAPI(title="LangGraph Agent API")

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: str

@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": f"{req.user_id}:{req.thread_id}"}}
    
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config
    )
    return {"response": result["messages"][-1].content}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    config = {"configurable": {"thread_id": f"{req.user_id}:{req.thread_id}"}}
    
    async def generate():
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/threads/{thread_id}/history")
async def get_history(thread_id: str, user_id: str):
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    state = graph.get_state(config)
    return {"messages": [str(m) for m in state.values.get("messages", [])]}
```

### 20.2 Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/langgraph
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: langgraph
    volumes:
      - pg_data:/var/lib/postgresql/data

volumes:
  pg_data:
```

### 20.3 Production Checklist

```
✅ Infrastructure
  □ PostgreSQL for checkpointing (not MemorySaver!)
  □ Redis for rate limiting / caching
  □ Containerized (Docker)
  □ Auto-scaling (Kubernetes or serverless)

✅ Observability
  □ LangSmith tracing enabled
  □ Structured logging (JSON)
  □ Metrics (latency, token usage, error rate)
  □ Alerts on failures

✅ Security
  □ API key rotation
  □ Input validation / sanitization
  □ Rate limiting per user
  □ Thread ID namespaced by user_id
  □ No secrets in code (use env vars)

✅ Reliability
  □ Retry logic for LLM API calls
  □ Fallback models (GPT-4o → GPT-4o-mini on failure)
  □ Timeout handling
  □ Max iteration limits in loops

✅ Cost Control
  □ Token counting / budgets
  □ Caching (LangChain cache layer)
  □ Model selection by task complexity
  □ Prompt optimization

✅ Testing
  □ Unit tests for each node
  □ Integration tests for full graph
  □ Evaluate outputs with LangSmith Evals
```

### 20.4 Caching LLM Calls

```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, SQLiteCache

# Development: in-memory cache
set_llm_cache(InMemoryCache())

# Production: SQLite or Redis cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

### 20.5 Rate Limiting & Budgets

```python
import tiktoken

def count_tokens(messages: list) -> int:
    enc = tiktoken.encoding_for_model("gpt-4o")
    total = sum(len(enc.encode(str(m))) for m in messages)
    return total

def token_guard_node(state: AgentState) -> dict:
    tokens = count_tokens(state["messages"])
    if tokens > 100_000:
        return {"error": "Token budget exceeded", "is_done": True}
    return {}  # No-op, continue
```

---

## 21. Best Practices

### 21.1 State Design

```python
# ✅ Keep state flat and typed
class GoodState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    status: str
    result: Optional[str]

# ❌ Avoid deeply nested or untyped state
class BadState(TypedDict):
    data: dict   # What's in here? No type safety!
```

### 21.2 Node Design

```python
# ✅ Single responsibility: one node = one job
def search_node(state): ...
def analyze_node(state): ...
def format_node(state): ...

# ❌ God node: does everything
def do_everything_node(state):
    # searches, analyzes, formats, saves, sends email...
    pass

# ✅ Return only what you changed
def good_node(state):
    return {"result": "value"}   # Partial update

# ❌ Return full state (breaks reducers)
def bad_node(state):
    return {**state, "result": "value"}   # Don't do this
```

### 21.3 Safety: Prevent Infinite Loops

```python
class SafeState(TypedDict):
    messages: Annotated[list, operator.add]
    iteration: int

def safety_check(state: SafeState) -> str:
    MAX_ITERATIONS = 10
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"⚠️ Max iterations ({MAX_ITERATIONS}) reached. Stopping.")
        return END
    return "agent"

def increment_node(state: SafeState) -> dict:
    return {"iteration": state["iteration"] + 1}
```

### 21.4 Thread Safety

```python
# ✅ Use thread_id to isolate user sessions
config = {"configurable": {"thread_id": f"user:{user_id}:session:{session_id}"}}

# ✅ Never share mutable state across threads
# ❌ Don't use global variables in nodes
```

### 21.5 Async Best Practices

```python
# ✅ Use async nodes for I/O-bound operations
async def async_search_node(state):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com?q={state['query']}")
        return {"results": resp.json()}

# ✅ Use ainvoke / astream for async execution
result = await graph.ainvoke(input_data, config=config)
async for chunk in graph.astream(input_data, config=config):
    print(chunk)
```

---

## 22. Glossary

| Term | Definition |
|---|---|
| **Agent** | An LLM that can use tools and make decisions autonomously |
| **Agentic AI** | AI systems that act autonomously over multiple steps |
| **Checkpointer** | Component that saves/loads graph state for persistence |
| **Compiled Graph** | A `StateGraph` after calling `.compile()` — ready to run |
| **Conditional Edge** | An edge that routes to different nodes based on state |
| **DAG** | Directed Acyclic Graph — a graph without cycles |
| **Edge** | A connection between two nodes |
| **END** | Built-in constant marking the graph's exit point |
| **Fan-out** | One node sending to multiple parallel nodes |
| **Fan-in** | Multiple nodes merging into one |
| **Graph** | The overall workflow structure of nodes + edges |
| **HITL** | Human-in-the-Loop — pausing for human review/approval |
| **Interrupt** | Pause execution at a specific node |
| **LLM** | Large Language Model (GPT-4o, Claude, Gemini, etc.) |
| **Memory** | Persisted state between graph runs |
| **MessagesState** | Built-in state with a `messages` list for chat |
| **Node** | A Python function that reads/writes to state |
| **ReAct** | Reasoning + Acting — LLM decides to use tools then acts |
| **Reducer** | Function that merges two values (used in Annotated state) |
| **Reflexion** | Pattern where agent critiques and revises its own output |
| **START** | Built-in constant marking the graph's entry point |
| **State** | Shared data object (TypedDict) passed between nodes |
| **StateGraph** | The main class for building graphs in LangGraph |
| **Streaming** | Receiving output token-by-token or chunk-by-chunk |
| **Subgraph** | A compiled graph used as a node inside another graph |
| **Supervisor** | Orchestrator agent that delegates to specialist agents |
| **Thread** | An isolated conversation/session identified by `thread_id` |
| **Tool** | A Python function the LLM can call (search, calculator, etc.) |
| **ToolNode** | Built-in node that executes tool calls from LLM responses |
| **tools_condition** | Built-in router: go to tools if there are tool calls, else END |

---

## 23. Resources & References

### Official Documentation
- 📘 [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- 📗 [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- 📙 [LangChain Docs](https://python.langchain.com/docs/introduction/)
- 🎓 [LangGraph Academy](https://academy.langchain.com/courses/intro-to-langgraph)

### Tutorials & Notebooks
- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [Multi-Agent Patterns](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
- [RAG with LangGraph](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)
- [LangSmith (Tracing)](https://docs.smith.langchain.com/)

### Community
- 💬 [LangChain Discord](https://discord.gg/langchain)
- 🐦 [LangChain Twitter/X](https://twitter.com/LangChainAI)
- 📺 [LangChain YouTube](https://www.youtube.com/@LangChain)

### Key Packages

```
langgraph              Core framework
langchain              LLM abstractions
langchain-openai       OpenAI integration
langchain-anthropic    Claude integration
langchain-google-genai Gemini integration
langchain-community    Community integrations
langgraph-cli          CLI for Studio + deployment
langsmith              Tracing and evaluation
```

---

## 🚀 Quick Reference Cheat Sheet

```python
# ─── IMPORTS ───────────────────────────────────────────
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, Optional
import operator

# ─── STATE ─────────────────────────────────────────────
class MyState(TypedDict):
    messages: Annotated[list, operator.add]   # appends
    status: str                                # replaces

# ─── NODES ─────────────────────────────────────────────
def my_node(state: MyState) -> dict:
    return {"status": "done"}                  # partial update only

# ─── BUILD GRAPH ────────────────────────────────────────
builder = StateGraph(MyState)
builder.add_node("node_a", node_a_fn)
builder.add_node("node_b", node_b_fn)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_conditional_edges("node_b", router_fn)

# ─── COMPILE ────────────────────────────────────────────
graph = builder.compile(checkpointer=MemorySaver())

# ─── RUN ────────────────────────────────────────────────
config = {"configurable": {"thread_id": "user-1"}}
result = graph.invoke({"messages": [...]}, config=config)

# ─── STREAM ─────────────────────────────────────────────
for chunk in graph.stream(input, config, stream_mode="updates"):
    print(chunk)

# ─── ASYNC ──────────────────────────────────────────────
result = await graph.ainvoke(input, config=config)

# ─── STATE INSPECTION ───────────────────────────────────
state = graph.get_state(config)
history = list(graph.get_state_history(config))

# ─── HUMAN IN LOOP ──────────────────────────────────────
graph = builder.compile(checkpointer=MemorySaver(),
                        interrupt_before=["sensitive_node"])
graph.invoke(input, config)          # Pauses before node
graph.update_state(config, updates)  # Human edits state
graph.invoke(None, config)           # Resume
```

---

*Made with ❤️ for AI builders. Happy graphing!*
