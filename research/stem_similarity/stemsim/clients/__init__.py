"""Clients for external services used by more than one part of the code.

Each module owns its service's endpoints, rate limits, error codes and retry
rules; callers keep only their own matching or business rules. `http` holds
the plumbing they share. A service used by a single module stays in that module.
"""
