# ai-research
Personal repo containing AI research and exploration

## Setup
**Note**: It is assumed that this is cloned into an environment that already has jupyter and pytorch setup; e.g., a "deeplearning-platform-release" GCE instance w/ jupyter setup on startup.

### Connecting to a VM
If interacting with this on a VM, users will likely need access to their Huggingface access tokens, git credentials and more.

1. Connect
```shell
gcloud compute ssh --project playground-dev-6ae7 --zone us-central1-a ai-dev-vm -- -L 8080:localhost:8080
```
2. Expose your tokens on the session
```shell
export GIT_USERNAME='${GIT_USERNAME}'
export GIT_TOKEN='${GIT_TOKEN}'
export HUGGINGFACE_TOKEN='${HUGGINGFACE_TOKEN}'
```
3. Use HTTPS to interact with git
```shell
git clone https://$GIT_USERNAME:$GIT_TOKEN@github.com/bradlet/ai-research.git
```

