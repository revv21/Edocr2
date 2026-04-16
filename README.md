flowchart TD
    USER([User]) --> FASTAPI[FastAPI Gateway]
    FASTAPI --> SUPERVISOR{LangGraph Supervisor}

    SUPERVISOR -->|rag| RAG
    SUPERVISOR -->|dfmea| DFMEA
    SUPERVISOR -->|optimize| OPT

    subgraph RAG [RAG Agent]
        R1[Retrieve Docs] --> R2[Grade Relevance]
        R2 -->|low confidence| R1
        R2 -->|sufficient| R3[Generate Answer]
    end

    subgraph DFMEA [DFMEA Agent]
        D1[Route Case] --> D2[RAG Context]
        D2 --> D3[Generate DFMEA Steps]
        D3 --> D4[Rate Risks & Export]
    end

    subgraph OPT [Optimiser Agent]
        O1[Solve ODE] --> O2[Compute RMS]
        O2 --> O3{Converged?}
        O3 -->|no| O1
        O3 -->|yes| O4[Summarise Result]
    end

    RAG & DFMEA & OPT -.-> INFRA

    subgraph INFRA [AWS Bedrock + FAISS]
        Claude[Claude Sonnet]
        Titan[Titan Embeddings]
        VectorStore[FAISS Vector Store]
    end
