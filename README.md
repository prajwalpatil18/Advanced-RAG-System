# 🚀 Advanced RAG (Retrieval-Augmented Generation) System

A **production-ready Advanced RAG system** designed for scalable, reliable, and efficient AI-powered question answering over custom documents. The project is **containerized with Docker** and **orchestrated using Kubernetes (K8s)** to support horizontal scaling and real-world workloads.

This repository is suitable for **enterprise-grade AI applications**, hackathons, and production deployments.

![CI](https://github.com/prajwalpatil18/Advanced-RAG-System/actions/workflows/ci.yml/badge.svg)

---

## ✨ Key Features

* 📄 **Document Ingestion with PDFs** 
* 🔍 **Vector-based Semantic Search** using embeddings
* 🧠 **LLM-powered Answer Generation** (RAG pipeline)
* ⚡ **Low-latency Retrieval** with chunking & metadata filtering
* 🐳 **Dockerized Application** for consistent deployments
* ☸️ **Kubernetes-ready** for scalability and high availability
* 📈 **Stateless API Design** – easy to scale horizontally
* 🔐 **Environment-based Configuration** for secrets & keys

---

## 🏗️ High-Level Architecture

```
User Query
   │
   ▼
API Service (FastAPI)
   │
   ├──► Retriever (Vector DB)
   │        └── Embeddings Store
   │
   └──► LLM (Generation)
            └── Context-aware Answer
```

Deployed as **containerized microservices**, managed via **Kubernetes Deployments & Services**.

---

## 🧰 Tech Stack

* **Backend**: Python, FastAPI
* **LLM Framework**: LangChain
* **Embeddings**: OpenAI / HuggingFace
* **Vector Store**: FAISS / Chroma / Pinecone *(configurable)*
* **Containerization**: Docker
* **Orchestration**: Kubernetes (K8s)
* **API Gateway**: FastAPI + Uvicorn
* **Cloud Ready**: AWS / GCP / Azure compatible

## 🐳 Docker Setup

### Build Image

```bash
docker build -t advanced-rag:latest .
```

### Run Container

```bash
docker run -p 8000:8000 --env-file .env advanced-rag:latest
```

API will be available at:

```
http://localhost:8000
```

---

## ☸️ Kubernetes Deployment

### Apply Kubernetes Manifests

```bash
kubectl apply -f k8s/
```

### Verify Deployment

```bash
kubectl get pods
kubectl get services
```

### Scale the Application

```bash
kubectl scale deployment advanced-rag --replicas=3
```

---

## ⚙️ Environment Variables

Create a `.env` file based on `.env.example`:

```
OPENAI_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-3-large
VECTOR_DB=faiss
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## 🔎 API Endpoints

### Health Check

```
GET /health
```

### Ask a Question

```
POST /chat
```

**Request Body**:

```json
{
  "query": "What is discussed in the uploaded document?"
}
```

**Response**:

```json
{
  "answer": "Generated answer based on retrieved context"
}
```

---

## 📈 Scalability & Production Readiness

* Stateless API → Horizontal Pod Autoscaling
* External Vector DB support
* ConfigMaps & Secrets for secure config
* Ready for Load Balancers & Ingress

---

## 🧪 Use Cases

* Enterprise Knowledge Base Q&A
* Healthcare / Legal Document Analysis
* Research Paper Search
* Internal Company Chatbots
* AI Assistants with Private Data

---

## 🛣️ Future Enhancements

* 🔐 Authentication & Role-based Access
* 📊 Monitoring (Prometheus + Grafana)
* 🧠 Fine-tuned Models Integration
* 🔄 Streaming Responses
* 🧾 Citation-based Answers

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Acknowledgements

* LangChain
* OpenAI / HuggingFace
* Kubernetes Community

---

If you find this project useful, don’t forget to ⭐ the repository!
