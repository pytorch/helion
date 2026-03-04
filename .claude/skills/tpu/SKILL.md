---
name: tpu
description: Context for working on Helion's TPU/Pallas backend. Auto-activate when the user mentions TPU or Pallas in the context of Helion development.
---

We are working on the Helion TPU/Pallas backend.

- Local pytorch checkout: ~/pytorch
- Local helion checkout: ~/helion
- TPU access: `kubectl exec -it <pod> -- /bin/bash` (default pod: `<username>-torchtpu`)
- On the TPU pod, activate the venv first: `source ~/.venv/bin/activate`
