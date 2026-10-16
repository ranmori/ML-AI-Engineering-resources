# AI agent memory system: semantic, episodic & procedural memory + RAG

## Architecture

```mermaid
flowchart TB
    subgraph SESSION["AI agent session — everything inside the box is ephemeral"]
        direction LR
        UP[User prompt]:::green
        CCH[Current chat history]
        SP[System prompt]
        UP --> WM[Working memory /<br/>Context RAM]
        CCH --> WM
        SP --> WM
        WM --> LLM((LLM — Q&A agent<br/>GPT, Claude)):::pink
        LLM --> REPLY[Reply]:::green
    end

    PROC["Procedural memory<br/>Files, text<br/>- How to act<br/>- Skills instructions<br/>(Skill.md)"]
    SEM["Semantic memory<br/>Vector store<br/>- Durable facts<br/>- User profile"]
    EPI["Episodic memory<br/>Vector store<br/>- Dated events<br/>- Past chat history"]

    PROC -->|"RAG (top-k search)"| WM
    SEM -->|"RAG (top-k search)"| WM
    EPI -->|"RAG (top-k search)"| WM

    LLM -->|"Save the messages / activities"| EPIDB[(Episodic vector DB)]
    SEM --> SEMDB[(Semantic vector DB)]
    EPI --> EPIDB

    EPIDB --> CONSOLIDATE{Only consolidate<br/>after N new chats}
    CONSOLIDATE --> SUMM((Summarizer agent<br/>cheaper models)):::pink
    SUMM -->|"Distill into facts"| SEMDB

    classDef green fill:#b2f2bb,stroke:#2f9e44,color:#1e1e1e;
    classDef pink fill:#ffc9c9,stroke:#e03131,color:#1e1e1e;
```

**How it works:**
- Everything inside the session box (user prompt, chat history, system prompt) is ephemeral — it exists only for that turn.
- Working memory pulls relevant context from three long-term stores via RAG top-k search: **procedural** (how to act — skills/instructions, e.g. `Skill.md`), **semantic** (durable facts, user profile), and **episodic** (dated events, past chat history).
- Every reply's raw messages/activity get saved into the episodic vector DB.
- Once N new chats accumulate, a cheaper "summarizer agent" consolidates episodic events, distilling them into durable facts that get written into the semantic vector DB.

*A couple of arrow directions/labels were inferred where the source frame was partly obscured by the video controls — flag anything that should read differently.*
