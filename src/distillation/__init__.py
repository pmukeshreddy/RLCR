"""Distillation pipeline: RL teacher → lightweight embedding model.

Phase 2 of the RLCR architecture. Converts the expensive DAPO-trained
scoring model into a fast sentence transformer that produces embeddings
where cosine similarity predicts the teacher's scores.
"""
