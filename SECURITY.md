# Security Policy

Lunaris is a scientific research codebase (lunar orbit propagation and the
ST-LRPS surrogate-gravity model), not a networked service. The most relevant
risks are therefore around untrusted inputs — model/kernel/dataset files and
configuration — rather than remote exploitation.

## Supported versions

The project is pre-1.0 and under active development. Security fixes are applied
to the latest `main`; there is no long-term support branch for older tags yet.

| Version | Supported |
| ------- | --------- |
| latest `main` / newest release | ✅ |
| older tagged releases | ❌ |

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** rather than opening a
public issue:

- Preferred: GitHub's private vulnerability reporting — "Report a vulnerability"
  under the repository's **Security** tab
  (<https://github.com/ayberkdt/lunaris/security>).
- Alternatively, email **ayberkdemirkanat@gmail.com** with a clear description, a
  minimal reproduction, and the affected version/commit.

Please do not disclose the issue publicly until a fix is available. We aim to
acknowledge a report within a few days and will keep you updated on remediation.

## Scope and handling notes

- Treat gravity-coefficient files, SPICE kernels, HDF5 datasets, and saved model
  checkpoints as **untrusted input** unless you produced them yourself. Loading a
  malicious file (e.g. a crafted pickle/checkpoint) can execute code — only load
  artifacts from sources you trust.
- Configuration and CLI arguments are likewise trusted-by-the-operator; Lunaris
  is intended to run locally or on HPC under the user's own account.
- Reports that amount to "loading an attacker-supplied pickle runs code" are a
  known property of the Python serialization formats involved; we still want to
  hear about avoidable cases (e.g. unnecessary deserialization of untrusted data).
