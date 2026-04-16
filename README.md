flowchart TD
    IN([Input: Component + Case]) --> ROUTER

    ROUTER{Case Type?}
    ROUTER -->|new design| RAG_CTX
    ROUTER -->|new environment| PARSE
    ROUTER -->|design change| PARSE

    PARSE[Parse Existing DFMEA\nxlsx import] --> RAG_CTX

    RAG_CTX[RAG Context Retrieval\nknown failures · historical ratings]

    RAG_CTX --> EL[Generate System Elements\nStep 1 — component hierarchy]
    EL --> FN[Generate Functions\nStep 4 — verb + noun format]
    FN --> FM[Generate Failure Modes & Effects\nStep 7 — severity S]
    FM --> FC[Generate Failure Causes\nStep 8 — occurrence O · detection D]
    FC --> RISK[Rate Action Priority\nStep 9 — H / M / L per AIAG-VDA 2019]
    RISK --> EXPORT[Assemble & Export\nformatted xlsx]
    EXPORT --> OUT([Output: DFMEA File])

    RAG_CTX -.->|context injected into\nall generation steps| FM
    RAG_CTX -.-> FC
