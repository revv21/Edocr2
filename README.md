flowchart TD

%% =====================
%% INPUT LAYER
%% =====================
A[Product Schematic\nParts & Subparts] --> B
A1[Function Definitions\nper Part] --> B
A2[Interfaces\nMechanical / Electrical / Fluid] --> B
A3[Manufacturing Process\n(Optional)] --> B
A4[Test Coverage Data\n(Optional)] --> B

%% =====================
%% DATA MODEL
%% =====================
B[System Graph Builder\n(Parts–Functions–Interfaces)] --> C

C[(Knowledge Graph\nPart ↔ Function ↔ Interface)] --> D
C --> E

%% =====================
%% FAILURE MODE GENERATION
%% =====================
D[Rule Engine\n(Function Negation,\nTemplates)] --> F
E[AI Engine\n(LLM + RAG)] --> F

F[Candidate Failure Modes\nper Function] --> G

%% =====================
%% EFFECT PROPAGATION
%% =====================
G --> H[Effect Propagation Engine\n(Graph Traversal)]
H --> I[Local Effects]
H --> J[Subsystem Effects]
H --> K[System / User Effects]

%% =====================
%% SCORING
%% =====================
K --> L[Severity Assignment\n(Rules + Standards)]
G --> M[Cause Inference Engine\n(Part Type + Process)]
M --> N[Occurrence Estimation\n(Priors / Data)]
A4 --> O[Detection Estimation\n(Test Coverage)]

L --> P
N --> P
O --> P

P[Risk Scoring\n(S / O / D → RPN)] --> Q

%% =====================
%% DFMEA OUTPUT
%% =====================
Q[DFMEA Item Generator] --> R[DFMEA Table\n(Function, FM, Effect, Cause,\nS/O/D, RPN)]

%% =====================
%% HUMAN LOOP
%% =====================
R --> S[Engineer Review & Edit]
S -->|Accepted| T[Approved DFMEA]
S -->|Modified| C

%% =====================
%% FTA GENERATION
%% =====================
T --> U[Fault Tree Builder]
U --> V[Top Event\n(System Failure)]
U --> W[Intermediate Events\n(Failure Modes)]
U --> X[Basic Events\n(Causes)]
V --> Y[Fault Tree Diagram\nAND / OR Gates]

%% =====================
%% LEARNING LOOP
%% =====================
S --> Z[Feedback Store]
Z --> E
Z --> D