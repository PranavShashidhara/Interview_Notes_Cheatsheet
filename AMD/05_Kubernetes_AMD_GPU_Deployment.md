# Kubernetes & vLLM Deployment on AMD GPUs

Build a production-ready vLLM inference server on Kubernetes with AMD Instinct GPUs.

## Course Objectives
1. Build single-node Kubernetes cluster as foundation for GPU-accelerated workloads
2. Install AMD GPU Operator — enable K8s to detect and schedule on AMD GPUs
3. Deploy vLLM + MetalLB — GPU-backed inference pods with load balancing and external IP
4. Test via API + Web UI — curl, Python client, browser chat demo
5. Monitor & troubleshoot — cluster internals and common deployment issues

## Architecture
```
Control Plane (API server, scheduler, controller)
    → Worker Node (GPU server, runs pods)
        → Container Runtime (containerd)
```

## Setup Steps

### 1. System Preparation
- Clear package locks that stall installation
- Verify RAM, disk space, network connectivity
- Confirm AMD Instinct GPU is present (`rocminfo`)

### 2. Install Kubernetes
- Orchestrates containerized workloads — the "air traffic controller" that decides where to run containers, how to store them, and how to recover when they fail
- Concepts scale from one node to hundreds — start simple, think big
- Control plane manages cluster state and workload distribution
- Worker node: GPU server runs pods with vLLM inference containers
- Container runtime (containerd with system groups): reliable container execution

### 3. AMD GPU Operator
| Component | Purpose |
|-----------|---------|
| Device Plugin | Exposes GPU resources to K8s scheduler |
| Node Labeller | Identifies GPU-capable nodes for placement |
| Driver Support | Installs/manages GPU drivers and monitoring |

### 4. Deploy vLLM + MetalLB
- vLLM: GPU-accelerated inference service in a K8s deployment
- MetalLB: software load balancer for bare-metal clusters, assigns real external IPs
- Flow: `Client → MetalLB → vLLM Pod`

### 5. Test & Validate
| Method | Purpose |
|--------|---------|
| `curl` to completions endpoint | Direct API validation — expect JSON responses with model-generated text choices |
| Python CLI client | Interactive client hitting API directly — demonstrates programmatic integration patterns |
| Web UI | Simple chat-style frontend providing friendly browser access to inference service |

Remote access: `ssh -L 8080:localhost:8080 user@server`

## Troubleshooting
| Symptom | Check | Fix |
|---------|-------|-----|
| Pod stuck Pending | `kubectl describe pod` for "insufficient amd.com/gpu" | Verify GPU Operator health |
| File not found / permission errors | PV mount status | Confirm PVC bound, check disk space |
| Connection refused on API | External IP assignment | Check MetalLB config + firewall |
| OOMKilled / CrashLoopBackOff | Container memory limits | Increase limits or reduce model size |

## Scaling & Next Steps
Production AI inference demands more than single-pod deployments. Scaling transforms generative AI from proof-of-concept to real-world impact.
- **Horizontal Pod Autoscaling**: auto-spawn replicas across GPU nodes on traffic surge
- **GPU sharing**: pack smaller inference jobs on shared GPUs for better utilization
- **Observability**: Prometheus + Grafana for metrics and dashboards
- **GitOps**: version-controlled deployments with safe rollbacks
- **Security**: RBAC, network policies, workload isolation for multi-tenant clusters
