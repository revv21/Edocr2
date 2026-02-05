flowchart TD

%% =====================
%% INPUT LAYER
%% =====================
A[Product Schematic\nParts and Subparts] --> B
A1[Function Definitions\nPer Part] --> B
A2[Interfaces\nMechanical Electrical Fluid] --> B
A3[Manufacturing Process\nOptional] --> B
A4[Test Coverage Data\nOptional] --> B

%% =====================
%% DATA MODEL
%% =====================
B[System Graph Builder\nParts Functions Interfaces] --> C

C[Knowledge Graph\nPart Function Interface] --> D
C --> E

%% =====================
%% FAILURE MODE GENERATION
%% =====================
D[Rule Engine\nFunction Negation Templates] --> F
E[AI Engine\nLLM and RAG] --> F

F[Candidate Failure Modes\nPer Function] --> G

%% =====================
%% EFFECT PROPAGATION
%% =====================
G --> H[Effect Propagation Engine]
H --> I[Local Effects]
H --> J[Subsystem Effects]
H --> K[System User Effects]

%% =====================
%% SCORING
%% =====================
K --> L[Severity Assignment\nStandards Based]
G --> M[Cause Inference Engine]
M --> N[Occurrence Estimation]
A4 --> O[Detection Estimation]

L --> P
N --> P
O --> P

P[Risk Scoring\nS O D to RPN] --> Q

%% =====================
%% DFMEA OUTPUT
%% =====================
Q[DFMEA Item Generator] --> R[DFMEA Table]

%% =====================
%% HUMAN LOOP
%% =====================
R --> S[Engineer Review]
S -->|Accepted| T[Approved DFMEA]
S -->|Modified| C

%% =====================
%% FTA GENERATION
%% =====================
T --> U[Fault Tree Builder]
U --> V[Top Event]
U --> W[Intermediate Events]
U --> X[Basic Events]
V --> Y[Fault Tree Diagram]

%% =====================
%% LEARNING LOOP
%% =====================
S --> Z[Feedback Store]
Z --> D
Z --> E