flowchart TD
    %% ─── Entry ───────────────────────────────────────────────────────────────
    USER(["`**User Request**
    RAG · DFMEA · Optimize`"])
    FASTAPI["`**FastAPI Gateway**
    /api/rag · /api/dfmea · /api/optimize
    SSE Streaming`"]
    SUPERVISOR{"`**Supervisor Node**
    reads: task_type`"}

    USER --> FASTAPI --> SUPERVISOR

    %% ─── Top-level routing ───────────────────────────────────────────────────
    SUPERVISOR -->|task_type = rag| RAG_START
    SUPERVISOR -->|task_type = dfmea| DFMEA_START
    SUPERVISOR -->|task_type = optimize| OPT_START
    SUPERVISOR -->|error| ERR

    ERR(["`⚠ Error Sink
    log + return`"])

    %% ═════════════════════════════════════════════════════════════════════════
    %% RAG SUB-GRAPH
    %% ═════════════════════════════════════════════════════════════════════════
    subgraph RAG ["　　　　　　　　　　　　 RAG Agent　　　　　　　　　　　　"]
        direction TB
        RAG_START["`**embed_query**
        Titan Embeddings V2`"]
        RETRIEVE["`**retrieve_docs**
        FAISS MMR  top-k=6`"]
        GRADE["`**grade_relevance**
        Claude Sonnet
        confidence score 0–1`"]
        REWRITE["`**rewrite_query**
        Claude Sonnet
        reformulate query`"]
        GENERATE["`**generate_answer**
        Claude Sonnet
        context-grounded reply`"]

        RAG_START --> RETRIEVE --> GRADE
        GRADE -->|confidence ≥ 0.7| GENERATE
        GRADE -->|confidence < 0.7| REWRITE
        REWRITE -->|re-retrieve cycle| RETRIEVE
        GENERATE --> RAG_END(["`✓ rag_answer
        rag_sources`"])
    end

    %% ═════════════════════════════════════════════════════════════════════════
    %% DFMEA SUB-GRAPH
    %% ═════════════════════════════════════════════════════════════════════════
    subgraph DFMEA ["　　　　　　　　　　　　 DFMEA Agent　　　　　　　　　　　　"]
        direction TB
        DFMEA_START["`**case_router**
        validate dfmea_case`"]
        CASE{"`dfmea_case?`"}
        PARSE["`**parse_import**
        dfmea_universal_parser
        extract elements · functions`"]
        RAG_CTX["`**dfmea_rag_context**  ★
        FAISS retrieval → Claude Sonnet
        known failures · historical S/O/D`"]
        GEN_EL["`**generate_elements**
        system hierarchy
        AIAG-VDA Step 1`"]
        GEN_FN["`**generate_functions**
        verb+noun functions
        AIAG-VDA Step 4`"]
        GEN_FM["`**generate_failures**
        failure modes + effects
        severity S  AIAG-VDA Step 7`"]
        GEN_FC["`**generate_causes**
        failure causes + controls
        occurrence O · detection D  Step 8`"]
        RATE["`**rate_risks**
        Action Priority H/M/L
        AIAG-VDA Step 9`"]
        ASSEMBLE["`**assemble_output**
        merge all steps → dfmea_output`"]
        EXPORT["`**export_xlsx**
        formatted AIAG-VDA workbook`"]

        DFMEA_START --> CASE
        CASE -->|new_env or design_change| PARSE
        CASE -->|new design| RAG_CTX
        PARSE --> RAG_CTX

        RAG_CTX -->|dfmea_rag_context injected into all prompts ↓| GEN_EL
        GEN_EL --> GEN_FN --> GEN_FM --> GEN_FC --> RATE --> ASSEMBLE --> EXPORT
        EXPORT --> DFMEA_END(["`✓ dfmea_output
        dfmea_export_path`"])
    end

    %% ═════════════════════════════════════════════════════════════════════════
    %% OPTIMIZER SUB-GRAPH
    %% ═════════════════════════════════════════════════════════════════════════
    subgraph OPT ["　　　　　　　　　　　　 Spring Optimizer Agent　　　　　　　　　　　　"]
        direction TB
        OPT_START["`**init_opt**
        k = midpoint of k_bounds
        create Optuna study`"]
        SOLVE["`**solve_ode**
        quarter-car 2-DOF model
        scipy RK45  x1 v1 x2 v2`"]
        RMS["`**compute_rms**
        RMS cabin acceleration
        update best k if improved`"]
        PROPOSE["`**propose_k**
        Optuna TPE sampler
        next k to evaluate`"]
        CONV{"`**check_convergence**
        ΔRMS < tol
        or iter ≥ max?`"}
        SUMM["`**summarize_opt**
        Claude Sonnet
        NVH engineering interpretation`"]

        OPT_START --> SOLVE --> RMS --> PROPOSE --> CONV
        CONV -->|not converged| SOLVE
        CONV -->|converged| SUMM
        SUMM --> OPT_END(["`✓ opt_best_k
        opt_best_rms
        opt_summary`"])
    end

    %% ═════════════════════════════════════════════════════════════════════════
    %% SHARED INFRASTRUCTURE
    %% ═════════════════════════════════════════════════════════════════════════
    subgraph INFRA ["　　　　　　　　　　　　 Shared Infrastructure　　　　　　　　　　　　"]
        direction LR
        BEDROCK["`**AWS Bedrock**
        Claude Sonnet  ap-south-1
        Titan Embeddings V2`"]
        FAISS_DB["`**FAISS Vector Store**
        Engineering docs
        chunk=512  overlap=64`"]
        REDIS["`**Redis Checkpointer**
        LangGraph state persistence
        resume on failure`"]
        LANGSMITH["`**LangSmith**
        Full trace · token cost
        node latencies`"]
    end

    %% ─── Infrastructure connections ──────────────────────────────────────────
    RAG -.->|LLM calls| BEDROCK
    DFMEA -.->|LLM calls| BEDROCK
    OPT -.->|summarize call| BEDROCK
    RAG -.->|vector search| FAISS_DB
    DFMEA -.->|vector search| FAISS_DB
    FASTAPI -.->|state snapshot| REDIS
    FASTAPI -.->|trace| LANGSMITH

    %% ─── Styling ─────────────────────────────────────────────────────────────
    classDef rag    fill:#0c2d48,stroke:#0ea5e9,color:#bae6fd
    classDef dfmea  fill:#1e1040,stroke:#8b5cf6,color:#ddd6fe
    classDef opt    fill:#052e16,stroke:#10b981,color:#a7f3d0
    classDef infra  fill:#1c1917,stroke:#57534e,color:#d6d3d1
    classDef gate   fill:#1c1400,stroke:#f59e0b,color:#fde68a
    classDef term   fill:#0f172a,stroke:#475569,color:#94a3b8

    class RAG_START,RETRIEVE,GRADE,REWRITE,GENERATE rag
    class DFMEA_START,CASE,PARSE,RAG_CTX,GEN_EL,GEN_FN,GEN_FM,GEN_FC,RATE,ASSEMBLE,EXPORT dfmea
    class OPT_START,SOLVE,RMS,PROPOSE,CONV,SUMM opt
    class BEDROCK,FAISS_DB,REDIS,LANGSMITH infra
    class FASTAPI,SUPERVISOR gate
    class USER,RAG_END,DFMEA_END,OPT_END,ERR term
