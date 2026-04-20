# Cortex — Master Technical Deep-Dive

## Master description

Cortex is a deployment-ready enterprise AI research platform built on an event-sourced CQRS architecture in Rust, with a React/Tauri frontend, GraphQL API gateway, multi-agent orchestration, RAG pipelines, and cloud-native Kubernetes infrastructure spanning 9 deployment phases. The system manages research workflows end-to-end — from natural language query to synthesized findings — using domain-driven design, multi-provider LLM integration, vector search, and deployment-ready MLOps infrastructure patterns.

The codebase is a little over 50,000 lines across Rust, TypeScript, and infrastructure-as-code. The backend implements proper event sourcing with aggregate roots, domain events with schema versioning and correlation tracking, an append-only PostgreSQL event store with snapshot support, CQRS command/query separation with read model projections, and a GraphQL API with real-time WebSocket subscriptions.

---

## 1) Problem framing

### 1.1 The business problem

Research workflows — market analysis, competitive intelligence, technology evaluation, security audits — are complex, multi-step processes that involve querying multiple sources, synthesizing findings, tracking provenance, and producing structured deliverables. Most teams manage this with ad-hoc documents, Slack threads, and manual hand-offs.

Cortex replaces that with a structured research platform where:

- a user submits a research query
- specialized AI agents (Researcher, Architect, Security, Analyst) execute parallel investigation phases
- findings are synthesized, quality-gated, and stored as first-class domain objects
- the entire history of every research workflow is preserved as an immutable event stream

### 1.2 Why event sourcing

Research workflows have three properties that make event sourcing the right architectural choice over CRUD:

1. **Auditability**: enterprise clients need to know exactly what happened, when, and why. An event log is a complete audit trail by construction.
2. **Temporal queries**: "What was the state of this research workflow at 3pm yesterday?" is trivial with event replay, impossible with CRUD.
3. **Workflow evolution**: research workflows go through multiple phases (created → running → tasks → completed/failed). Modeling this as state transitions via events is natural.

### 1.3 Why Rust

The backend is written in Rust, not Node/Python/Go, for three reasons:

- **Correctness via the type system**: Rust gave me strong type safety for domain models and event contracts, while invalid workflow transitions were enforced at the domain layer through typed validation and errors.
- **Performance**: Rust's zero-cost abstractions and lack of GC make it suitable for high-throughput event processing and low-latency API responses.
- **Concurrency safety**: `Arc<RwLock<...>>` and Rust's ownership model prevent data races in the concurrent event store, projection system, and multi-agent coordinator without runtime overhead.

---

## 2) Event sourcing foundation (Phase 4.1)

### 2.1 Core abstractions

The event store is built on four core Rust traits and types:

**`DomainEvent` trait**:

```rust
#[async_trait]
pub trait DomainEvent: Debug + Send + Sync {
    fn event_type(&self) -> &'static str;
    fn event_version(&self) -> u32 { 1 }
    fn correlation_id(&self) -> Option<Uuid> { None }
    fn causation_id(&self) -> Option<Uuid> { None }
    fn serialize(&self) -> Result<serde_json::Value, serde_json::Error>;
    fn validate(&self) -> Result<(), String> { Ok(()) }
}
```

Every domain event carries:
- `event_type`: string discriminator for deserialization
- `event_version`: schema version for forward-compatible evolution
- `correlation_id`: groups related events across aggregates
- `causation_id`: tracks which event caused this event (causal chain)

**`AggregateRoot` trait**:

```rust
#[async_trait]
pub trait AggregateRoot: Debug + Send + Sync {
    type Event: DomainEvent + Clone;
    type State: Clone + Send + Sync + Serialize + Deserialize;

    fn get_id(&self) -> AggregateId;
    fn get_version(&self) -> u64;
    fn get_uncommitted_events(&self) -> &[Self::Event];
    fn mark_events_as_committed(&mut self);
    fn apply_event(&mut self, event: &Self::Event);
    fn get_state(&self) -> &Self::State;
    fn restore_from_state(id: AggregateId, state: Self::State, version: u64) -> Self;
    fn validate(&self) -> EventStoreResult<()> { Ok(()) }
}
```

This is the standard aggregate root pattern: the aggregate holds uncommitted events, applies them to its internal state, and exposes a snapshotable state for persistence optimization.

**`EventStore`**:

PostgreSQL-backed, with:
- append-only event streams keyed by `stream_id` (UUID)
- optimistic concurrency control via `expected_version`
- transactional multi-event appends
- an internal `EventBus` for dispatching events to projections

**`SnapshotStore`**:

Pluggable via the `SnapshotStorage` trait. Snapshots are taken every N events (configurable, default 100) to avoid replaying the entire event history on every aggregate load.

### 2.2 Research workflow events

The `ResearchWorkflowEvent` enum defines 7 event variants:

```rust
pub enum ResearchWorkflowEvent {
    WorkflowCreated { workflow_id, name, query, methodology, created_at, correlation_id },
    ExecutionStarted { workflow_id, started_at, correlation_id },
    TaskCreated { workflow_id, task_id, task_type, agent_type, created_at, correlation_id },
    TaskCompleted { workflow_id, task_id, results, completed_at, correlation_id },
    ExecutionCompleted { workflow_id, results, completed_at, correlation_id },
    ExecutionFailed { workflow_id, error, failed_at, correlation_id },
    WorkflowUpdated { workflow_id, updates, updated_at, correlation_id },
}
```

Each variant carries a `correlation_id: Option<Uuid>` for distributed tracing. Extracting this from every variant would require 7 match arms. Instead, a custom macro eliminates that boilerplate:

```rust
macro_rules! impl_correlation_id {
    ($enum_name:ident, $($variant:ident),+ $(,)?) => {
        fn correlation_id(&self) -> Option<Uuid> {
            match self {
                $( $enum_name::$variant { correlation_id, .. } => *correlation_id, )+
            }
        }
    };
}
```

This keeps the pattern DRY as new event variants are added. It is a small detail but demonstrates Rust macro fluency.

### 2.3 Aggregate state machine

The `ResearchWorkflowAggregate` enforces valid state transitions at the domain level:

```
Created  →  Running  →  Completed
                    →  Failed
         →  Failed (can fail from Created too, e.g., validation failure)
```

Each command method validates the current state before emitting an event:

- `create_workflow`: validates name/query non-empty, emits `WorkflowCreated`
- `start_execution`: requires status == `Created`, emits `ExecutionStarted`
- `create_task`: requires status == `Running`, validates task_type non-empty, emits `TaskCreated`
- `complete_task`: requires task exists and is not already completed, emits `TaskCompleted`
- `complete_execution`: requires status == `Running`, emits `ExecutionCompleted`
- `fail_execution`: requires status is not already `Completed` or `Failed`, emits `ExecutionFailed`

Invalid transitions return typed `EventStoreError::InvalidOperation` errors — not panics, not HTTP 500s.

### 2.4 Event store concurrency

The `append_events` method uses optimistic concurrency:

```rust
pub async fn append_events(
    &self,
    stream_id: StreamId,
    events: Vec<Box<dyn DomainEvent>>,
    expected_version: Option<u64>,
) -> Result<u64, EventStoreError> {
    let mut tx = self.pool.begin().await?;

    if let Some(expected) = expected_version {
        let current_version = self.get_stream_version_tx(&mut tx, &stream_id).await?;
        if current_version != expected {
            return Err(EventStoreError::ConcurrencyConflict {
                expected,
                actual: current_version,
            });
        }
    }
    // ... serialize and insert events within transaction
}
```

If two concurrent commands try to modify the same aggregate, the second one gets a `ConcurrencyConflict` error. The caller can retry with a fresh aggregate load. This is the standard optimistic locking pattern for event stores.

### 2.5 Event replay

The `EventReplayService` supports rebuilding aggregates and projections from the event log:

- configurable batch size (default 100 events per batch)
- concurrent stream replay (up to 10 streams in parallel)
- checkpointing every N events to support resume after failure
- progress tracking (total/processed streams, total/processed events, failed streams)
- snapshot-aware: can load the latest snapshot and replay only events after it

This is critical for two operational scenarios:
1. rebuilding read models after a schema change
2. recovering from a corrupted projection

### 2.6 Snapshot store

Snapshots serialize the aggregate's full state to JSON and store it alongside a version number:

```rust
pub struct Snapshot {
    pub stream_id: Uuid,
    pub snapshot_version: u64,
    pub snapshot_data: serde_json::Value,
    pub snapshot_metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}
```

The `SnapshotStorage` trait is pluggable — the primary implementation is `PostgresSnapshotStorage`, but the trait allows swapping in Redis or S3 for different deployment profiles. Snapshot frequency is configurable (default: every 100 events). The snapshot interval is a tradeoff between replay cost and extra write/storage overhead; the default of 100 events was chosen as a practical middle ground.

Why PostgreSQL for the event store: PostgreSQL was a good fit because I needed transactional appends, optimistic concurrency, JSON payload storage, mature indexing, and operational simplicity. At this stage, correctness and developer velocity mattered more than inventing a custom event-store backend.

---

## 3) CQRS pattern (Phase 4.2)

### 3.1 Command side

Commands are separate types that implement the `Command` trait:

```rust
pub trait Command: Send + Sync + Debug {
    fn validate(&self) -> CQRSResult<()> { Ok(()) }
    fn command_name(&self) -> &'static str;
    fn command_id(&self) -> Uuid;
    fn correlation_id(&self) -> Option<Uuid> { None }
}
```

6 command types: `CreateResearchWorkflow`, `StartWorkflowExecution`, `CreateTask`, `CompleteTask`, `CompleteWorkflow`, `FailWorkflow`.

Each command handler follows the same load-mutate-save pattern. To eliminate duplication, the `AggregatePersistence` struct encapsulates the shared logic:

```rust
struct AggregatePersistence {
    event_store: Arc<EventStore>,
}

impl AggregatePersistence {
    async fn load(&self, workflow_id: Uuid) -> CQRSResult<ResearchWorkflowAggregate> {
        let events = self.event_store.read_events(workflow_id, None, None).await?;
        // Rebuild aggregate from events
    }

    async fn save(&self, aggregate: &mut ResearchWorkflowAggregate) -> CQRSResult<u64> {
        let uncommitted = aggregate.get_uncommitted_events();
        let expected_version = aggregate.get_version() - uncommitted.len() as u64;
        self.event_store.append_events(aggregate.get_id(), events, Some(expected_version)).await?;
        aggregate.mark_events_as_committed();
    }
}
```

Every command handler composes `AggregatePersistence` instead of re-implementing the event store round-trip. This is a key design decision that keeps the handler code focused purely on business logic.

The `CommandBus` dispatches commands to handlers with middleware support (validation, logging, metrics).

### 3.2 Query side

Queries implement a `Query` trait with built-in caching support:

```rust
pub trait Query: Send + Sync + Debug {
    type Result: Send + Sync;
    fn is_cacheable(&self) -> bool { true }
    fn cache_key(&self) -> String;
    fn cache_ttl_seconds(&self) -> u64 { 300 }
}
```

5 query types: `GetResearchWorkflow`, `GetWorkflowList` (paginated), `GetWorkflowStats`, `GetTasksByWorkflow`, `SearchWorkflows` (full-text search with status/date filters).

Queries read from **read models** — denormalized, query-optimized views — not from the event store directly. This is the core CQRS benefit: the write side (event store + aggregates) is optimized for consistency, while the read side (projections + read models) is optimized for query performance. The read side is eventually consistent with the write side, because projections process appended events asynchronously. That is the normal CQRS tradeoff: stronger write-side consistency, faster query-side performance.

### 3.3 Read models

```rust
pub struct ResearchWorkflowReadModel {
    pub id: Uuid,
    pub name: String,
    pub query: String,
    pub methodology: Option<serde_json::Value>,
    pub status: WorkflowStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub results: Option<serde_json::Value>,
    pub tasks: Vec<TaskReadModel>,
    pub metrics: WorkflowMetrics,
    pub tags: Vec<String>,
}
```

Read models include pre-computed `WorkflowMetrics` (total_tasks, completed_tasks, progress_percentage, estimated_completion_time) that would be expensive to compute on every query from raw events.

The `ReadModelStore` trait abstracts storage — in-memory for tests, PostgreSQL for production.

### 3.4 Projections

Projections are event handlers that update read models asynchronously. The `ProjectionBuilder` trait:

```rust
#[async_trait]
pub trait ProjectionBuilder: Send + Sync {
    fn projection_name(&self) -> &str;
    fn supported_event_types(&self) -> Vec<&'static str>;
    async fn handle_event(&self, stream_id: Uuid, event: &dyn DomainEvent,
                          read_model_store: Arc<RwLock<dyn ReadModelStore>>) -> CQRSResult<()>;
    async fn reset(&self, read_model_store: Arc<RwLock<dyn ReadModelStore>>) -> CQRSResult<()>;
}
```

The `ProjectionManager` orchestrates multiple projections with:
- checkpoint tracking per projection (last processed event ID, sequence, timestamp)
- error counting and status tracking (Active, Paused, Error, Rebuilding, Stopped)
- configurable batch size and checkpoint frequency
- reset capability for full projection rebuilds

### 3.5 Full test suite

Both the event store and CQRS modules include dedicated `tests.rs` files with unit tests covering:
- aggregate creation and state transitions
- invalid state transition rejection
- optimistic concurrency conflict detection
- event serialization round-trips
- command handler execution
- projection event processing
- read model query correctness

---

## 4) GraphQL API gateway (Phase 4.4)

### 4.1 Technology

Built with `async-graphql` (Rust) + `axum` HTTP server. The schema type:

```rust
pub type GraphQLSchema = Schema<QueryRoot, MutationRoot, SubscriptionRoot>;
```

### 4.2 Application context

Every resolver has access to a shared `AppContext` with 10 service dependencies:

```rust
pub struct AppContext {
    pub database: Arc<dyn DatabaseService>,
    pub event_store: Arc<dyn EventStoreService>,
    pub cqrs_service: Arc<dyn CQRSService>,
    pub auth_service: Arc<dyn AuthService>,
    pub api_manager: Arc<dyn ApiManagerService>,
    pub research_engine: Arc<dyn ResearchEngineService>,
    pub bmad_service: Arc<dyn BMadService>,
    pub cache: Arc<dyn CacheService>,
    pub metrics: Arc<dyn MetricsService>,
    pub config: Arc<AppConfig>,
}
```

All services are behind `Arc<dyn Trait>` — interface-driven, testable, swappable.

### 4.3 Key API features

- **DataLoader pattern**: batches N+1 queries into single bulk loads. Without DataLoader, fetching 50 workflows with tasks would issue 50 separate task queries. With DataLoader, it issues 1.
- **Real-time subscriptions**: WebSocket-based live updates for workflow progress. When a task completes, all subscribed clients receive the update immediately.
- **Rate limiting**: per-user and per-API-key throttling (configurable requests_per_minute and burst_size).
- **Auth middleware**: per-resolver authentication with admin-level gating.
- **Query depth/complexity limits**: prevents malicious deep-nested queries from overwhelming the server (configurable max_query_depth, max_query_complexity).
- **Federation support**: schema registry integration for federated GraphQL if the platform scales to multiple backend services.

---

## 5) Multi-agent orchestration

### 5.1 Agent types

4 specialized agents, each with distinct research capabilities:

| Agent | Role | Research Types |
|---|---|---|
| Researcher | Primary investigation | Market, technology, competitive |
| Architect | System design analysis | Architecture patterns, infrastructure |
| Security | Compliance research | Security, compliance, privacy |
| Analyst | Data-driven insights | User research, analytics |

### 5.2 Workflow coordination

The `WorkflowCoordinationService` orchestrates multi-agent execution:

- maintains `active_workflows: Arc<RwLock<HashMap<Uuid, ActiveWorkflow>>>` for concurrent workflow tracking
- each workflow progresses through phases: Initializing → ResearchPhase → SynthesisPhase → DocumentationPhase → ValidationPhase → Completed
- agents can execute in parallel within a phase
- each workflow tracks cost (`WorkflowCostTracking`) and quality (`WorkflowQualityMetrics`)

### 5.3 Quality metrics

Per-workflow quality tracking:

```rust
pub struct WorkflowQualityMetrics {
    pub overall_confidence: f64,
    pub research_coverage: f64,
    pub source_diversity: f64,
    pub evidence_completeness: f64,
    pub agent_consensus: f64,
    pub quality_gates_passed: u32,
    pub quality_gates_total: u32,
}
```

Quality gates must be passed before a workflow advances to the next phase. This prevents low-confidence or incomplete research from being presented as final output.

### 5.4 Research bridge

The `ResearchBridgeService` (905 lines) integrates BMAD agents with the Cortex research engine. It handles:

- translating BMAD research requests into Cortex workflow commands
- routing to the appropriate agent based on `ResearchType`
- supporting 4 research methodologies (DonLim, NickScamara, Hybrid, Comprehensive)
- 4 depth levels (Basic, Standard, Comprehensive, Expert)
- cost limits and duration limits per research request

### 5.5 Multi-provider LLM support

The platform integrates 6 LLM providers: OpenAI, Hugging Face, Groq, Together AI, Replicate, and Ollama (local). A cost-optimized routing layer selects the provider based on query complexity, latency requirements, and cost constraints.

---

## 6) RAG pipeline (Phase 5.0)

### 6.1 Components

Deployed as Kubernetes services:

- **Qdrant** (v1.11.0): vector database for high-performance similarity search with persistent volume claims
- **Embedding service**: generates semantic document embeddings (OpenAI or Hugging Face)
- **Document processor**: ingests, chunks, and indexes research documents
- **RAG service**: retrieval-augmented generation — retrieves relevant chunks, injects into LLM prompt
- **MCP server**: Model Context Protocol for standardized AI model communication
- **Ollama**: local LLM inference with GPU resource allocation for privacy-preserving workloads
- **Cost optimizer + model router**: intelligent routing by cost/quality/latency tradeoff

### 6.2 How RAG integrates with the research workflow

When an agent executes a research task:

1. Query → embedding service → vector similarity search in Qdrant
2. Top-K relevant document chunks retrieved
3. Chunks injected into the LLM prompt as context
4. LLM generates a grounded response with source attribution
5. Results stored as a `TaskCompleted` event with full provenance

---

## 7) Kubernetes infrastructure (9 phases)

### 7.1 Phase overview

| Phase | System | Key Technologies |
|---|---|---|
| 4.1 | Event Sourcing | PostgreSQL event store, Rust aggregates |
| 4.2 | CQRS | Command/query buses, projections, read models |
| 4.3 | Kubernetes + Istio | Service mesh, mTLS, multi-zone HA |
| 4.4 | GraphQL API Gateway | async-graphql, WebSocket subscriptions, DataLoader |
| 4.5 | Serverless Edge | Knative functions, edge computing |
| 4.6 | MLOps | Kubeflow Pipelines, MLflow, TensorFlow Serving, GPU |
| 4.7 | Analytics | ClickHouse, Kafka, Airflow |
| 4.8 | Enterprise Auth | Multi-tenant Keycloak SSO (SAML, OAuth2, MFA), RBAC, billing |
| 4.9 | Security & Compliance | Vault, Velero, Falco, SOC 2/GDPR/HIPAA frameworks |
| 5.0 | AI Enhancement | RAG, Qdrant, Ollama, MCP, cost optimization |

Each phase has its own deployment shell script (`deploy-phase-X.Y.sh`) with prerequisite checks, namespace setup, manifest application, and health validation.

### 7.2 MLOps pipeline (Phase 4.6)

I defined and integrated MLOps infrastructure patterns — Kubeflow, MLflow, TensorFlow Serving, and model monitoring — as part of the platform architecture, rather than building a custom model-training workflow end to end inside Cortex itself.

- **Kubeflow Pipelines** (v1.8.0): DAG-based ML workflow orchestration
- **MLflow** (v2.8.1): model registry, experiment tracking, versioning, A/B testing
- **TensorFlow Serving** (v2.14.0): GPU-accelerated inference (<100ms P95 latency target, architecturally supported by async inference pipeline, request batching, and Istio sidecar load balancing)
- **Model monitoring**: drift detection, performance degradation alerts

### 7.3 Analytics stack (Phase 4.7)

- **ClickHouse**: columnar data warehouse for petabyte-scale analytics
- **Apache Kafka**: streaming event ingestion from the event store
- **Apache Airflow**: ETL orchestration for automated data pipelines
- **Grafana + Prometheus + Jaeger**: dashboards, metrics, distributed tracing

### 7.4 Security (Phase 4.9)

- **Zero-trust**: mTLS everywhere via Istio, Kubernetes network policies, Falco runtime protection
- **HashiCorp Vault**: centralized secrets management and rotation
- **Velero**: disaster recovery with 4-hour RTO, 1-hour RPO
- **Compliance frameworks**: SOC 2, GDPR, HIPAA

---

## 8) Frontend

### 8.1 Web application

React 18 + TypeScript + Tailwind + Vite. Features:

- Research workflow creation and monitoring
- Real-time workflow progress via GraphQL subscriptions
- Agent activity visualization
- Research results display with source attribution

### 8.2 Desktop application

Tauri-based cross-platform desktop app wrapping the same React frontend with native OS integration. Includes Tauri command bindings (`tauri_commands.rs`) that bridge the Rust backend directly to the desktop UI without going through HTTP.

---

## 9) Code quality and testing

### 9.1 Test coverage

Both `event_store/tests.rs` (17,628 bytes) and `cqrs/tests.rs` (16,263 bytes) contain comprehensive unit tests:

- aggregate lifecycle tests (create → start → task → complete)
- invalid state transition tests (start a completed workflow → error)
- concurrency conflict tests
- serialization round-trip tests
- command handler tests
- query handler tests with read model assertions
- projection processing tests

### 9.2 Error handling

Structured error types with categories:

```rust
pub enum ErrorCategory {
    Validation,
    NotFound,
    Concurrency,
    Serialization,
    Storage,
    Internal,
}
```

Every error carries context (operation name, entity ID, details) for debugging. No `unwrap()` in production paths.

---

## 10) Technology stack summary

| Layer | Technologies |
|---|---|
| Backend | Rust, tokio, serde, async-trait, sqlx, uuid, chrono, axum |
| Event Store | PostgreSQL 15 (append-only), optimistic concurrency, snapshots |
| API | async-graphql, axum, WebSocket subscriptions, DataLoader |
| Frontend | React 18, TypeScript, Tailwind, Vite, Redux Toolkit |
| Desktop | Tauri (Rust + WebView) |
| AI/ML | Kubeflow, MLflow, TensorFlow Serving, Qdrant, Ollama, MCP |
| Streaming | Apache Kafka, Apache Airflow |
| Auth | Multi-tenant Keycloak SSO (SAML, OAuth2, MFA), RBAC, JWT |
| Infra | Docker, Kubernetes, Istio, Knative, Helm |
| Security | HashiCorp Vault, Falco, Velero, mTLS |
| Monitoring | Prometheus, Grafana, Jaeger |
| Cache | Redis 7 |

---

## 11) What was NOT implemented — do not claim

- The Rust code compiles as structured source files but the full binary was not deployed to a live Kubernetes cluster with real traffic
- TensorFlow Serving, Kubeflow, ClickHouse, Kafka, Airflow, Keycloak, Vault, Velero, Falco are defined as Kubernetes manifests and deployment scripts — they were not load-tested with live production traffic
- The 50,000+ concurrent users capacity is an architecture-supported design target based on Kubernetes HPA auto-scaling, Istio load balancing, and multi-zone HA — it was not measured under synthetic or production load, so present it as "the infrastructure is architected to support 50K+ concurrent users" rather than "we tested with 50K users"
- The <100ms P95 inference latency is architecturally supported by TensorFlow Serving's async request batching, GPU acceleration, and Istio sidecar proxy routing — present it as "the inference pipeline targets <100ms P95" rather than "we measured <100ms P95"
- The compliance frameworks (SOC 2, GDPR, HIPAA) are infrastructure patterns, not formal certifications

---

## 12) Ownership

I architected and implemented:

- **Event-sourced CQRS backend**: the `AggregateRoot` trait, `DomainEvent` trait, `EventStore` with optimistic concurrency, `SnapshotStore`, `EventReplayService`, the `impl_correlation_id!` macro, the `AggregatePersistence` abstraction, all command/query handlers, projections, and read models
- **Multi-agent orchestration**: the `WorkflowCoordinationService`, `ResearchBridgeService`, `AgentEnhancementService`, quality metrics tracking, cost tracking, and the 4-agent research workflow
- **GraphQL API layer**: the full async-graphql schema with subscriptions, DataLoader, rate limiting, auth middleware, and federation support
- **RAG pipeline integration**: Qdrant deployment, embedding service, document processor, MCP server, cost-optimized model routing
- **Kubernetes infrastructure**: all 9 deployment phases including Istio service mesh, MLOps, analytics, enterprise auth, and security/compliance manifests
- **Frontend**: React web app and Tauri desktop app with real-time subscription integration

This was a solo project. I did not write the underlying frameworks (async-graphql, axum, sqlx, Kubeflow, etc.) — I used them as building blocks.

---

## 13) Defensible vs must-qualify

### Fully defensible

- Event sourcing with Rust aggregate roots and domain events — the code is in the repo
- CQRS with full command/query separation, projections, and read models
- Optimistic concurrency control on the event store
- The `impl_correlation_id!` macro and `AggregatePersistence` abstraction
- GraphQL API with DataLoader, subscriptions, rate limiting
- Multi-agent orchestration with quality gates and cost tracking
- 9-phase Kubernetes infrastructure with deployment scripts
- RAG pipeline with Qdrant vector search
- 50,000+ lines of code across Rust, TypeScript, and IaC
- Full test suites for event store and CQRS

### Must qualify

- Infrastructure targets (50K+ concurrent users, 99.9% uptime, <100ms P95) are architecture-supported design targets — the infrastructure patterns (Kubernetes HPA, Istio service mesh, multi-zone HA, TensorFlow Serving GPU batching) support these numbers, but they were not validated under synthetic or production load
- Kubernetes manifests are deployment-ready but were not run under production load
- Compliance frameworks are patterns, not formal certifications
- MLOps and analytics stacks are defined as infrastructure, not trained/deployed ML models

---

## 14) Strongest spoken version

"I built Cortex, a deployment-ready enterprise AI research platform — the codebase was a little over 50,000 lines across Rust, TypeScript, and infrastructure-as-code. The core architecture is event-sourced CQRS in Rust. The event store is PostgreSQL-backed with append-only streams, optimistic concurrency control, and a pluggable snapshot store that avoids replaying the full event history on every aggregate load. I defined aggregate roots using a Rust trait with generic associated types for events and state, and the research workflow aggregate enforces valid state transitions at the domain layer — invalid transitions like completing a workflow that hasn't been started return typed InvalidOperation errors rather than producing inconsistent state.

On top of that I built a full CQRS layer with separate command and query buses, read model projections, and checkpoint-based replay for rebuilding projections. To avoid duplicating the load-mutate-save pattern across all command handlers, I created an AggregatePersistence abstraction that each handler composes. The query side reads from denormalized read models with pre-computed metrics rather than replaying events on every query.

The API layer is a GraphQL gateway built with async-graphql and axum, with DataLoader for N+1 prevention, WebSocket subscriptions for real-time workflow updates, per-user rate limiting, and query depth and complexity limits.

For the AI layer, I built a multi-agent orchestration system with four specialized agents — researcher, architect, security, and analyst — that execute in parallel within workflow phases. Each workflow tracks quality metrics like confidence, source diversity, and evidence completeness, with quality gates that must pass before advancing to the next phase. The research pipeline integrates a RAG system using Qdrant for vector search, multi-provider LLM support with cost-optimized routing, and an MCP server for standardized model communication.

The infrastructure spans 9 Kubernetes deployment phases, each with its own deployment script and validation flow, covering service mesh, serverless, MLOps, analytics, auth, and security."

---

## 15) Exact phrases to use under pressure

On event sourcing: "Every state change is persisted as an immutable domain event. The aggregate root replays its event history to reconstruct current state, with snapshots every 100 events to avoid full replay on every load. This gives me a complete audit trail, temporal queries, and the ability to rebuild any read model from scratch."

On why PostgreSQL for the event store: "PostgreSQL was a good fit because I needed transactional appends, optimistic concurrency, JSON payload storage, mature indexing, and operational simplicity. At this stage, correctness and developer velocity mattered more than inventing a custom event-store backend."

On snapshot frequency: "Snapshot frequency is a tradeoff between replay cost and write amplification. Every 100 events was a reasonable default that reduced aggregate load time without making snapshot writes dominate event-store throughput."

On read model consistency: "The read side is eventually consistent with the event store, because projections update asynchronously after event append. That is the normal CQRS tradeoff: stronger write-side consistency, faster query-side performance."

On optimistic concurrency: "The event store uses optimistic concurrency — when appending events, I pass the expected aggregate version. If another command modified the aggregate concurrently, I get a ConcurrencyConflict error and the caller retries with a fresh load."

On CQRS: "Commands go through the event store and aggregates. Queries read from denormalized read models that are updated asynchronously by projections. The write side is optimized for consistency; the read side is optimized for query performance. If someone asks whether the read models are strongly consistent — no, they are eventually consistent, which is the standard CQRS tradeoff."

On why CQRS is worth the complexity: "Because the write side and read side had different optimization goals. The write side needed correctness, auditability, and workflow-state consistency through aggregates and events. The read side needed fast UI queries, filtering, and precomputed workflow metrics. CQRS let me keep those concerns separate instead of forcing one data model to serve both."

On AggregatePersistence: "Every command handler needs to load the aggregate, apply a domain method, and save the uncommitted events. I extracted that into a shared AggregatePersistence struct so each handler only contains business logic, not event store plumbing."

On the correlation ID macro: "Each event variant carries a correlation_id field. Instead of writing 7 match arms to extract it, I wrote a macro that generates the extraction for all variants. It's a small thing but it keeps the code DRY as new events are added."

On GraphQL: "I used async-graphql with DataLoader for N+1 prevention, WebSocket subscriptions for real-time updates, and per-user rate limiting. Query depth and complexity limits prevent abusive queries."

On multi-agent orchestration: "Four specialized agents execute in parallel within workflow phases. Quality gates — confidence, coverage, evidence completeness — must pass before advancing. Each workflow also tracks cost so we can optimize model routing."

On infrastructure: "I defined 9 deployment phases, each with its own script. The infrastructure is architected for 50K+ concurrent users using Kubernetes HPA auto-scaling, Istio service mesh with multi-zone HA, and TensorFlow Serving for <100ms P95 inference. These are architecture-supported targets based on the infrastructure patterns I deployed, though they were not validated under synthetic load."

On multi-tenant auth: "Multi-tenant Keycloak SSO handles enterprise authentication with SAML and OAuth2 federation, MFA enforcement, and RBAC policies that scope every API call, event stream, and research workflow to the tenant's organizational boundary. This is critical for enterprise clients where cross-tenant data isolation is a contractual and compliance requirement."

---

## 16) Exact phrases to avoid

- "This runs in production with 50,000 concurrent users" (architecture-supported target, not measured under load)
- "We are SOC 2 / GDPR / HIPAA certified" (infrastructure patterns, not formal certification)
- "The system achieves 99.9% uptime" (target, not measured SLA)
- "I wrote async-graphql / Kubeflow / Qdrant" (I used them as building blocks)
- "The ML models are trained and deployed" (MLOps infrastructure is defined, models are not custom-trained)
- "I benchmarked <100ms inference latency" (architecture-supported by TensorFlow Serving GPU batching and Istio routing, not formally benchmarked on this system)
