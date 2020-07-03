"""Orion Evaluation subpackage.

This subpackage contains all the code related to the
Orion Evaluation usage.
"""
from orion.evaluation.contextual import (
    contextual_accuracy, contextual_confusion_matrix, contextual_f1_score, contextual_precision,
    contextual_recall)
from orion.evaluation.point import (
    point_accuracy, point_confusion_matrix, point_f1_score, point_precision, point_recall)

CONTEXTUAL_METRICS = {
    'accuracy': contextual_accuracy,
    'f1': contextual_f1_score,
    'recall': contextual_recall,
    'precision': contextual_precision,
}

POINT_METRICS = {
    'accuracy': point_accuracy,
    'f1': point_f1_score,
    'recall': point_recall,
    'precision': point_precision,
}

__all__ = (
    'contextual_accuracy', 'contextual_confusion_matrix', 'contextual_f1_score',
    'contextual_precision', 'contextual_recall', 'CONTEXTUAL_METRICS',
    'point_accuracy', 'point_confusion_matrix', 'point_f1_score', 'point_precision',
    'point_recall', 'POINT_METRICS'
)
