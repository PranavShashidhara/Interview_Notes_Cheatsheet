# Docker and Kubernetes

## Docker

### Core Concepts

**Image**: read-only template with application code, dependencies, and OS layers. Built from Dockerfile.
**Container**: running instance of an image. Isolated process with own filesystem, network, process namespace.
**Layer**: each Dockerfile instruction creates a read-only layer. Images share layers via copy-on-write.
**Registry**: image storage (Docker Hub, ECR, GCR, GHCR).

### Dockerfile (Your ONNX/FastAPI MLOps Project)
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching: only rebuild if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### .dockerignore (Your Resume)
Prevents unnecessary files from being sent to build context:
```
__pycache__/
*.pyc
.git/
.env
*.log
tests/
.pytest_cache/
node_modules/
```

### Essential Docker Commands
```bash
# Build
docker build -t myapp:latest .
docker build -t myapp:v1.0 --build-arg ENV=prod .

# Run
docker run -d -p 8000:8000 --name myapp myapp:latest
docker run -e API_KEY=secret -v /host/data:/app/data myapp:latest
docker run --rm myapp:latest pytest  # ephemeral container for testing

# Inspect
docker ps                         # running containers
docker ps -a                      # all containers
docker logs myapp -f              # follow logs
docker exec -it myapp /bin/bash   # interactive shell
docker inspect myapp              # detailed info

# Images
docker images
docker pull python:3.11-slim
docker push myrepo/myapp:latest
docker tag myapp:latest myrepo/myapp:v1.0

# Cleanup
docker stop myapp && docker rm myapp
docker system prune -a            # remove all unused resources
```

### Docker Compose (Multi-service Development)
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      - db
    volumes:
      - ./models:/app/models

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Multi-stage Build (Minimize Image Size)
```dockerfile
# Stage 1: build
FROM python:3.11 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: runtime (slim)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["python", "app.py"]
```

### Networking
- **bridge**: default; containers on same host can communicate via container name
- **host**: share host network namespace; fastest; no isolation
- **overlay**: span multiple Docker hosts; used in Swarm/Kubernetes
- **none**: no networking

### Volume Types
- **Named volumes**: `docker volume create mydata`; managed by Docker; persist across container restarts
- **Bind mounts**: `-v /host/path:/container/path`; maps host directory; useful for development
- **tmpfs**: in-memory; not persisted; for sensitive data

---

## Testing with Docker (Your pytest + FastAPI Project)
```python
# conftest.py
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)

# test_api.py
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_valid(client):
    response = client.post("/predict", json={"features": [1, 0, 22.0, 1, 0, 7.25, 0]})
    assert response.status_code == 200
    assert "prediction" in response.json()
```

Key point: TestClient runs ASGI app directly without needing live server — CI/CD friendly.

---

## Kubernetes (K8s)

### Architecture

**Control Plane**:
- **API Server**: REST frontend; all components communicate through it
- **etcd**: distributed key-value store; cluster state
- **Scheduler**: assigns pods to nodes based on resources/constraints
- **Controller Manager**: runs controllers (ReplicaSet, Deployment, etc.)

**Worker Nodes**:
- **kubelet**: agent on each node; ensures containers run per Pod spec
- **kube-proxy**: maintains network rules; service routing
- **Container Runtime**: Docker, containerd, CRI-O

### Core Objects

#### Pod
Smallest deployable unit. One or more containers sharing network namespace and storage.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-api
  labels:
    app: ml-api
spec:
  containers:
  - name: api
    image: myrepo/ml-api:v1.0
    ports:
    - containerPort: 8000
    resources:
      requests:
        memory: "512Mi"
        cpu: "250m"
      limits:
        memory: "1Gi"
        cpu: "500m"
    env:
    - name: MODEL_PATH
      value: "/models/model.onnx"
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 5
```

#### Deployment
Manages ReplicaSet; handles rolling updates and rollbacks.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: myrepo/ml-api:v1.0
```

#### Service
Stable network endpoint for pods (pods are ephemeral; IPs change).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer   # ClusterIP (internal), NodePort (dev), LoadBalancer (cloud)
```

#### ConfigMap and Secret
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  MODEL_NAME: "lightgbm-v1"
  LOG_LEVEL: "INFO"

---
apiVersion: v1
kind: Secret
metadata:
  name: db-creds
type: Opaque
data:
  password: <base64-encoded>
```

#### Ingress
Routes external HTTP/HTTPS to services. Requires Ingress controller (NGINX, Traefik).

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
spec:
  rules:
  - host: api.mycompany.com
    http:
      paths:
      - path: /predict
        pathType: Prefix
        backend:
          service:
            name: ml-api-service
            port:
              number: 80
```

### kubectl Commands
```bash
# Apply manifests
kubectl apply -f deployment.yaml
kubectl apply -f manifests/

# View resources
kubectl get pods
kubectl get pods -o wide           # with node info
kubectl get deployments
kubectl get services
kubectl describe pod ml-api-xxx    # detailed events and state

# Debugging
kubectl logs ml-api-xxx
kubectl logs ml-api-xxx -c sidecar  # specific container
kubectl exec -it ml-api-xxx -- /bin/bash
kubectl port-forward svc/ml-api-service 8080:80

# Scaling
kubectl scale deployment ml-api --replicas=5
kubectl autoscale deployment ml-api --min=2 --max=10 --cpu-percent=70

# Rolling update and rollback
kubectl set image deployment/ml-api api=myrepo/ml-api:v2.0
kubectl rollout status deployment/ml-api
kubectl rollout undo deployment/ml-api
kubectl rollout history deployment/ml-api
```

### HorizontalPodAutoscaler (HPA)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Namespaces
Logical cluster partitioning. Production, staging, dev environments share same cluster.
```bash
kubectl create namespace production
kubectl get pods -n production
kubectl config set-context --current --namespace=production
```

### PersistentVolume for ML Models
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
spec:
  accessModes:
    - ReadWriteMany     # multiple pods can read (model serving)
  resources:
    requests:
      storage: 10Gi
  storageClassName: efs-sc  # EFS for RWX on AWS
```

---

## CI/CD with Docker and Kubernetes

### GitHub Actions Pipeline
```yaml
name: ML API CI/CD
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: docker build -t test-image . && docker run test-image pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Build and push
      run: |
        docker build -t $ECR_REGISTRY/ml-api:$GITHUB_SHA .
        docker push $ECR_REGISTRY/ml-api:$GITHUB_SHA
    - name: Deploy to K8s
      run: |
        kubectl set image deployment/ml-api api=$ECR_REGISTRY/ml-api:$GITHUB_SHA
        kubectl rollout status deployment/ml-api
```

---

## Interview Key Points

**Docker**
- Docker vs VM: containers share host OS kernel; VMs have separate OS; containers are lighter, faster to start, less isolated
- What is a layer? Each Dockerfile instruction creates an immutable layer. Layers are cached and shared across images. Changes only rebuild from the modified instruction onward.
- What is the difference between CMD and ENTRYPOINT? ENTRYPOINT defines the executable; CMD provides default arguments. CMD can be overridden at runtime; ENTRYPOINT is harder to override.
- Why multi-stage builds? Separate build environment (with compilers, dev deps) from runtime image (lean). Results in much smaller production image.

**Kubernetes**
- Pod vs Deployment: Pod is a single instance; Deployment manages multiple identical pods, handles restarts, rolling updates, scaling.
- How does K8s handle a failed pod? Kubelet detects crash; restarts container per restartPolicy. If node fails, controller manager recreates pods on healthy nodes.
- What is a Service and why is it needed? Pods are ephemeral with changing IPs. Service provides stable DNS name + IP, load balances across matching pods.
- Rolling update vs Blue-Green: Rolling: gradually replace old pods with new; zero downtime but both versions coexist temporarily. Blue-Green: full parallel environment; instant switch; requires 2x resources.
- How does HPA work? Metrics server collects CPU/memory; HPA controller compares to target; adjusts replica count. Custom metrics via Prometheus adapter also supported.
