# How to design an AI agent system for e-commerce DTC customer support

**Scope:** Returns, exchanges, WISMO (where-is-my-order)

## Goals
- ≥70% automation
- CSAT > 4.5/5
- p50 < 1s, p95 < 2.5s

## Rules
- Return window: <30 days only, item must be returned in good condition
- Refund policy: <$50 auto-approved, ≥$50 requires manager approval
- Exchange policy: good condition only, within eligible product categories
- Route to human: when asked by the user, or on emotion detection

## Backend / DB
- Sized for peak RPS and seasonality
- Data storage: relational DB

## Architecture

```mermaid
flowchart TB
    subgraph OBS[" "]
        direction TB
        OBSLABEL[Observability, metrics & evals]
    end

    RDB[(Relational DB)] -->|"User auth, fetch chats/emails"| UC[User channels<br/>Web chat, email]
    UC <-->|Auth| GW{Gateway<br/>Auth/SSO, PII,<br/>rate limit, dedupe}
    UC -->|"Respond to the user<br/>in real time"| CRM[CRM<br/>Write/fetch pipeline,<br/>create/update tickets]

    UC -->|"Ask a question:<br/>'I want to return product X'"| RA((Router agent<br/>Understands user intent,<br/>decides agent routing))
    HA[Human approval] <--> RA
    RA <--> QA((Q&A agent<br/>Speaks to user<br/>at all times))

    RA -->|"Decide to take the<br/>return route action"| RPA((Return planner agent<br/>Decides functional calling<br/>or a deterministic workflow))
    RPA -->|"Return latest status back<br/>to Q&A agent to talk to user"| QA

    VDB[(Vector DB<br/>FAQs, policies)] -->|"RAG: check policy<br/>via vector DB"| RPA
    VDB -->|"Align with policy"| RPA

    RPA -->|"Inform the agent"| PG{Policy / guardrails<br/>Capability tokens, RBAC,<br/>human approval,<br/>read-back confirmation}

    PG -->|"If it fits the policy,<br/>call Shopify API for return"| SHOP[Shopify API<br/>Track order status, RMA,<br/>exchanges]
    SHOP --> STRIPE[Stripe API<br/>Payments / refunds]
    STRIPE -->|"Payment refund done"| RPA
```

*A few arrow labels were inferred where the handwriting was small or cut off in the source frame — flag anything that should read differently.*
