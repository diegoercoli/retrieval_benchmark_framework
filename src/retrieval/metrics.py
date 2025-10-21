import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd

from src.preprocessing.preprocess_dataset import GroundTruthDocumentRecord
from src.core.configuration import SearchType


@dataclass
class QueryEvaluationMetrics:
    """Metrics for evaluating a single query's retrieval performance"""
    query_id: str
    query_text: str

    # Document-level metrics (filename match only)
    document_precision: float
    document_recall: float
    document_f1_score: float
    document_ndcg: float
    document_mrr: float
    document_map: float  # NEW: Mean Average Precision at document level


    # Section-level metrics (exact chapter/section/subsection match)
    section_precision: float
    section_recall: float
    section_f1_score: float
    section_ndcg: float
    section_mrr: float
    section_map: float  # NEW: Mean Average Precision at section level


    search_type: SearchType
    configuration_id: str
    ground_truth_count: int
    retrieved_count_total: int  # Always 50 chunks retrieved
    retrieved_count_evaluated: int  # top_k chunks used for P/R

    # Counts for document level
    document_relevant_retrieved_count: int

    # Counts for section level
    section_relevant_retrieved_count: int


class RetrievalMetricsCalculator:
    """Calculator for various retrieval evaluation metrics at both document and section levels"""

    @staticmethod
    def calculate_dual_level_map(
            retrieved_chunks: List[dict],
            ground_truth: List[GroundTruthDocumentRecord],
            k: int = 50
    ) -> Tuple[float, float]:
        """
        Calculate Mean Average Precision (MAP) at both document and section levels.

        MAP measures the quality of ranked retrieval results by computing the average
        of precision values at each position where a relevant document is retrieved.

        Formula:
            MAP = (1/R) * Σ(Precision@k * rel(k))
            where R is the total number of relevant documents
            and rel(k) is 1 if the kth result is relevant, 0 otherwise

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries (should be 50)
            ground_truth: List of ground truth document records
            k: Number of chunks to consider for MAP (default 50 for ranking evaluation)

        Returns:
            Tuple of (document_map, section_map)

        Example:
            If we have 3 relevant documents and retrieve them at positions [1, 3, 5]:
            - Precision@1 = 1/1 = 1.0
            - Precision@3 = 2/3 = 0.667
            - Precision@5 = 3/5 = 0.6
            - MAP = (1.0 + 0.667 + 0.6) / 3 = 0.756
        """
        if not retrieved_chunks or not ground_truth:
            return 0.0, 0.0

        k = min(k, len(retrieved_chunks))

        # Create ground truth sets for both levels
        gt_documents = set(gt.filename for gt in ground_truth)
        gt_sections = set()
        for gt in ground_truth:
            section_info = RetrievalMetricsCalculator._extract_gt_section_info(gt)
            if section_info:
                gt_sections.add((gt.filename, section_info))

        # Track precision at each relevant position
        doc_precisions_at_relevant = []
        sec_precisions_at_relevant = []

        doc_relevant_count = 0
        sec_relevant_count = 0

        # Iterate through retrieved chunks and calculate precision at relevant positions
        for i, chunk in enumerate(retrieved_chunks[:k]):
            filename = chunk.get('filename', '')
            section_info = RetrievalMetricsCalculator._extract_section_info(chunk)

            # Document level
            if filename in gt_documents:
                doc_relevant_count += 1
                # Precision at this position
                doc_precision_at_i = doc_relevant_count / (i + 1)
                doc_precisions_at_relevant.append(doc_precision_at_i)

            # Section level
            if section_info and (filename, section_info) in gt_sections:
                sec_relevant_count += 1
                # Precision at this position
                sec_precision_at_i = sec_relevant_count / (i + 1)
                sec_precisions_at_relevant.append(sec_precision_at_i)

        # Calculate MAP: average of precisions at relevant positions
        # Normalize by total number of relevant documents (not retrieved count)
        doc_map = sum(doc_precisions_at_relevant) / len(gt_documents) if gt_documents else 0.0
        sec_map = sum(sec_precisions_at_relevant) / len(gt_sections) if gt_sections else 0.0

        return doc_map, sec_map

    @staticmethod
    def calculate_dual_level_metrics(
            retrieved_chunks: List[dict],
            ground_truth: List[GroundTruthDocumentRecord],
            top_k: int = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Calculate precision, recall, and F1-score at both document and section levels.
        Uses only the top_k chunks for precision/recall calculation.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries (should be 50 chunks)
            ground_truth: List of ground truth document records
            top_k: Number of top chunks to use for precision/recall (from configuration)

        Returns:
            Tuple of ((doc_precision, doc_recall, doc_f1), (sec_precision, sec_recall, sec_f1))
        """
        if not retrieved_chunks:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        if not ground_truth:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        # Use only top_k chunks for precision/recall calculation
        if top_k is not None:
            eval_chunks = retrieved_chunks[:top_k]
        else:
            eval_chunks = retrieved_chunks

        # Extract retrieved information from top_k chunks only
        retrieved_documents = set()  # filename only
        retrieved_sections = set()  # (filename, section_info)

        for chunk in eval_chunks:
            filename = chunk.get('filename', '')
            if filename:
                retrieved_documents.add(filename)

                # Try to get the most specific section information available
                section_info = RetrievalMetricsCalculator._extract_section_info(chunk)
                if section_info:
                    retrieved_sections.add((filename, section_info))

        # Extract ground truth information
        ground_truth_documents = set()  # filename only
        ground_truth_sections = set()  # (filename, section_info)

        for gt_record in ground_truth:
            ground_truth_documents.add(gt_record.filename)

            # Use the most specific section information from ground truth
            section_info = RetrievalMetricsCalculator._extract_gt_section_info(gt_record)
            if section_info:
                ground_truth_sections.add((gt_record.filename, section_info))

        # Calculate document-level metrics
        doc_relevant_retrieved = retrieved_documents.intersection(ground_truth_documents)
        doc_precision = len(doc_relevant_retrieved) / len(retrieved_documents) if retrieved_documents else 0.0
        doc_recall = len(doc_relevant_retrieved) / len(ground_truth_documents) if ground_truth_documents else 0.0
        doc_f1 = (2 * doc_precision * doc_recall) / (doc_precision + doc_recall) if (
                                                                                                doc_precision + doc_recall) > 0 else 0.0

        # Calculate section-level metrics
        sec_relevant_retrieved = retrieved_sections.intersection(ground_truth_sections)
        sec_precision = len(sec_relevant_retrieved) / len(retrieved_sections) if retrieved_sections else 0.0
        sec_recall = len(sec_relevant_retrieved) / len(ground_truth_sections) if ground_truth_sections else 0.0
        sec_f1 = (2 * sec_precision * sec_recall) / (sec_precision + sec_recall) if (
                                                                                                sec_precision + sec_recall) > 0 else 0.0

        return (doc_precision, doc_recall, doc_f1), (sec_precision, sec_recall, sec_f1)

    @staticmethod
    def _extract_section_info(chunk: dict) -> str:
        """Extract the most specific section information from a chunk"""
        # Priority: subsection > section > chapter
        if chunk.get('subsection'):
            return chunk['subsection']
        elif chunk.get('section'):
            return chunk['section']
        elif chunk.get('chapter'):
            return chunk['chapter']
        return ""

    @staticmethod
    def _extract_gt_section_info(gt_record: GroundTruthDocumentRecord) -> str:
        """Extract section information from ground truth record"""
        # Use section_title which should contain the most specific information
        return gt_record.section_title if gt_record.section_title else ""

    @staticmethod
    def calculate_dual_level_ndcg(
            retrieved_chunks: List[dict],
            ground_truth: List[GroundTruthDocumentRecord],
            k: int = 50
    ) -> Tuple[float, float]:
        """
        Calculate NDCG at both document and section levels using all 50 chunks.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries (should be 50)
            ground_truth: List of ground truth document records
            k: Number of chunks to consider for NDCG (default 50 for ranking evaluation)

        Returns:
            Tuple of (document_ndcg, section_ndcg)
        """
        if not retrieved_chunks or not ground_truth:
            return 0.0, 0.0

        k = min(k, len(retrieved_chunks))

        # Create ground truth lookups with confidence-based relevance scores
        gt_doc_relevance = {}  # filename -> relevance
        gt_sec_relevance = {}  # (filename, section_info) -> relevance

        for gt_record in ground_truth:
            # Document level
            relevance = gt_record.confidence.value
            if gt_record.filename in gt_doc_relevance:
                gt_doc_relevance[gt_record.filename] = max(gt_doc_relevance[gt_record.filename], relevance)
            else:
                gt_doc_relevance[gt_record.filename] = relevance

            # Section level
            section_info = RetrievalMetricsCalculator._extract_gt_section_info(gt_record)
            if section_info:
                key = (gt_record.filename, section_info)
                if key in gt_sec_relevance:
                    gt_sec_relevance[key] = max(gt_sec_relevance[key], relevance)
                else:
                    gt_sec_relevance[key] = relevance

        # Calculate DCG for both levels using all k chunks
        doc_dcg = 0.0
        sec_dcg = 0.0

        for i, chunk in enumerate(retrieved_chunks[:k]):
            filename = chunk.get('filename', '')
            section_info = RetrievalMetricsCalculator._extract_section_info(chunk)

            # Document level DCG
            doc_relevance = gt_doc_relevance.get(filename, 0)
            if doc_relevance > 0:
                doc_dcg += doc_relevance / math.log2(i + 2)

            # Section level DCG
            sec_relevance = gt_sec_relevance.get((filename, section_info), 0)
            if sec_relevance > 0:
                sec_dcg += sec_relevance / math.log2(i + 2)

        # Calculate IDCG for both levels
        doc_ideal_relevances = sorted(gt_doc_relevance.values(), reverse=True)[:k]
        sec_ideal_relevances = sorted(gt_sec_relevance.values(), reverse=True)[:k]

        doc_idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(doc_ideal_relevances))
        sec_idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(sec_ideal_relevances))

        # Calculate NDCG
        doc_ndcg = doc_dcg / doc_idcg if doc_idcg > 0 else 0.0
        sec_ndcg = sec_dcg / sec_idcg if sec_idcg > 0 else 0.0

        return doc_ndcg, sec_ndcg

    @staticmethod
    def calculate_dual_level_mrr(
            retrieved_chunks: List[dict],
            ground_truth: List[GroundTruthDocumentRecord]
    ) -> Tuple[float, float]:
        """
        Calculate Mean Reciprocal Rank (MRR) at both document and section levels.
        Uses all retrieved chunks to find the first relevant result.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries (should be 50)
            ground_truth: List of ground truth document records

        Returns:
            Tuple of (document_mrr, section_mrr)
        """
        if not retrieved_chunks or not ground_truth:
            return 0.0, 0.0

        # Create ground truth sets
        gt_documents = set(gt.filename for gt in ground_truth)
        gt_sections = set()
        for gt in ground_truth:
            section_info = RetrievalMetricsCalculator._extract_gt_section_info(gt)
            if section_info:
                gt_sections.add((gt.filename, section_info))

        # Find first relevant result for each level
        doc_mrr = 0.0
        sec_mrr = 0.0
        doc_found = False
        sec_found = False

        for i, chunk in enumerate(retrieved_chunks):
            filename = chunk.get('filename', '')
            section_info = RetrievalMetricsCalculator._extract_section_info(chunk)

            # Document level MRR - first relevant document
            if not doc_found and filename in gt_documents:
                doc_mrr = 1.0 / (i + 1)  # Rank is 1-indexed
                doc_found = True

            # Section level MRR - first relevant section
            if not sec_found and section_info and (filename, section_info) in gt_sections:
                sec_mrr = 1.0 / (i + 1)  # Rank is 1-indexed
                sec_found = True

            # Early termination if both found
            if doc_found and sec_found:
                break

        return doc_mrr, sec_mrr

    @staticmethod
    def count_relevant_chunks_dual_level(
            chunks: List[dict],
            ground_truth: List[GroundTruthDocumentRecord],
            top_k: int = None
    ) -> Tuple[int, int]:
        """
        Count relevant chunks at both document and section levels.
        Uses only top_k chunks for counting (matching precision/recall evaluation).

        Args:
            chunks: List of retrieved chunks (should be 50)
            ground_truth: List of ground truth records
            top_k: Number of top chunks to count (from configuration)

        Returns:
            Tuple of (document_level_count, section_level_count)
        """
        if not chunks or not ground_truth:
            return 0, 0

        # Use only top_k chunks for counting (same as precision/recall)
        if top_k is not None:
            eval_chunks = chunks[:top_k]
        else:
            eval_chunks = chunks

        # Create ground truth sets
        gt_documents = set(gt.filename for gt in ground_truth)
        gt_sections = set()
        for gt in ground_truth:
            section_info = RetrievalMetricsCalculator._extract_gt_section_info(gt)
            if section_info:
                gt_sections.add((gt.filename, section_info))

        # Count matches in top_k chunks only
        doc_count = 0
        sec_count = 0

        for chunk in eval_chunks:
            filename = chunk.get('filename', '')

            # Document level
            if filename in gt_documents:
                doc_count += 1

            # Section level
            section_info = RetrievalMetricsCalculator._extract_section_info(chunk)
            if section_info and (filename, section_info) in gt_sections:
                sec_count += 1

        return doc_count, sec_count



class MetricsAggregator:
    """Aggregates and analyzes metrics across multiple queries and configurations"""

    def __init__(self):
        self.metrics: List[QueryEvaluationMetrics] = []

        # ADD THESE NEW METHODS:

    def get_distinct_query_count(self) -> int:
        """
        Get the count of distinct queries in the dataset.

        Returns:
            Number of unique query IDs
        """
        unique_query_ids = set(metric.query_id for metric in self.metrics)
        return len(unique_query_ids)

    def get_distinct_queries(self) -> set:
        """
        Get the set of distinct query IDs.

        Returns:
            Set of unique query IDs
        """
        return set(metric.query_id for metric in self.metrics)

    def add_metrics(self, metrics: QueryEvaluationMetrics):
        """Add query evaluation metrics to the aggregator"""
        self.metrics.append(metrics)

    def get_metrics_by_configuration(self) -> Dict[str, List[QueryEvaluationMetrics]]:
        """Group metrics by configuration ID"""
        config_metrics = {}
        for metric in self.metrics:
            config_id = metric.configuration_id
            if config_id not in config_metrics:
                config_metrics[config_id] = []
            config_metrics[config_id].append(metric)
        return config_metrics

    def get_metrics_by_search_type(self) -> Dict[SearchType, List[QueryEvaluationMetrics]]:
        """Group metrics by search type"""
        search_type_metrics = {}
        for metric in self.metrics:
            search_type = metric.search_type
            if search_type not in search_type_metrics:
                search_type_metrics[search_type] = []
            search_type_metrics[search_type].append(metric)
        return search_type_metrics

    def calculate_average_metrics_by_config(self) -> Dict[str, Dict[str, float]]:
        """Calculate average metrics for each configuration at both levels"""
        config_metrics = self.get_metrics_by_configuration()
        averages = {}

        for config_id, metrics_list in config_metrics.items():
            if not metrics_list:
                continue

            # Document level averages
            avg_doc_precision = sum(m.document_precision for m in metrics_list) / len(metrics_list)
            avg_doc_recall = sum(m.document_recall for m in metrics_list) / len(metrics_list)
            avg_doc_f1 = sum(m.document_f1_score for m in metrics_list) / len(metrics_list)
            avg_doc_ndcg = sum(m.document_ndcg for m in metrics_list) / len(metrics_list)
            avg_doc_mrr = sum(m.document_mrr for m in metrics_list) / len(metrics_list)
            avg_doc_map = sum(m.document_map for m in metrics_list) / len(metrics_list)  # NEW

            # Section level averages
            avg_sec_precision = sum(m.section_precision for m in metrics_list) / len(metrics_list)
            avg_sec_recall = sum(m.section_recall for m in metrics_list) / len(metrics_list)
            avg_sec_f1 = sum(m.section_f1_score for m in metrics_list) / len(metrics_list)
            avg_sec_ndcg = sum(m.section_ndcg for m in metrics_list) / len(metrics_list)
            avg_sec_mrr = sum(m.section_mrr for m in metrics_list) / len(metrics_list)
            avg_sec_map = sum(m.section_map for m in metrics_list) / len(metrics_list)  # NEW

            averages[config_id] = {
                # Document level
                'document_precision': avg_doc_precision,
                'document_recall': avg_doc_recall,
                'document_f1_score': avg_doc_f1,
                'document_ndcg': avg_doc_ndcg,
                'document_mrr': avg_doc_mrr,
                'document_map': avg_doc_map,  # NEW
                # Section level
                'section_precision': avg_sec_precision,
                'section_recall': avg_sec_recall,
                'section_f1_score': avg_sec_f1,
                'section_ndcg': avg_sec_ndcg,
                'section_mrr': avg_sec_mrr,
                'section_map': avg_sec_map,  # NEW
                # General
                'query_count': len(metrics_list)
            }

        return averages

    # ==============================================================================
    # STEP 5: Update to_dataframe to include MAP
    # ==============================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all metrics to a pandas DataFrame for analysis"""
        data = []
        for metric in self.metrics:
            data.append({
                'query_id': metric.query_id,
                'query_text': metric.query_text,
                'configuration_id': metric.configuration_id,
                'search_type': metric.search_type.name,
                'ground_truth_count': metric.ground_truth_count,
                'retrieved_count_total': metric.retrieved_count_total,
                'retrieved_count_evaluated': metric.retrieved_count_evaluated,

                # Document level
                'document_precision': metric.document_precision,
                'document_recall': metric.document_recall,
                'document_f1_score': metric.document_f1_score,
                'document_ndcg': metric.document_ndcg,
                'document_mrr': metric.document_mrr,
                'document_map': metric.document_map,  # NEW
                'document_relevant_retrieved': metric.document_relevant_retrieved_count,

                # Section level
                'section_precision': metric.section_precision,
                'section_recall': metric.section_recall,
                'section_f1_score': metric.section_f1_score,
                'section_ndcg': metric.section_ndcg,
                'section_mrr': metric.section_mrr,
                'section_map': metric.section_map,  # NEW
                'section_relevant_retrieved': metric.section_relevant_retrieved_count
            })

        return pd.DataFrame(data)

    def get_best_configuration_by_metric(self, metric_name: str = 'section_f1_score') -> tuple[str, float]:
        """
        Find the best configuration based on a specific metric.

        Args:
            metric_name: Name of the metric to optimize for (e.g., 'document_f1_score', 'section_precision')

        Returns:
            Tuple of (configuration_id, average_metric_value)
        """
        averages = self.calculate_average_metrics_by_config()

        if not averages:
            return None, 0.0

        best_config = max(averages.items(), key=lambda x: x[1][metric_name])
        return best_config[0], best_config[1][metric_name]

    def clear(self):
        """Clear all stored metrics"""
        self.metrics.clear()