"""
Shared services module for Research Core and Synvo Profiling.

This module provides a unified service factory that can create embedding,
vectordb, and LLM services from a shared configuration file.
"""

from src.shared.service_factory import SharedServiceFactory

__all__ = ["SharedServiceFactory"]
