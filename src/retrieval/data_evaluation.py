import json
from typing import List

from src.core.configuration import RAGConfiguration, SearchConfiguration, EvaluationConfiguration
#from src.preprocessing.preprocess_dataset import process_dataset, GroundTruthDocumentRecord
from src.retrieval.metrics import (
    RetrievalMetricsCalculator,
    MetricsAggregator,
    QueryEvaluationMetrics, GroundTruthDocumentRecord
)


class EvaluationService:
    """
    Service responsible for evaluating retrieval results against ground truth.
    Follows SRP by focusing solely on evaluation logic with dual-level metrics.

    Evaluates at two levels:
    - Document Level: True positive if chunk belongs to correct filename
    - Section Level: True positive if chunk matches exact chapter/section/subsection

    Retrieval Strategy:
    - Always retrieves 50 chunks for comprehensive ranking evaluation
    - Uses configured top_k for precision/recall calculations
    - Uses all 50 chunks for NDCG and MRR calculations
    """

    def __init__(self, evaluation_configuration: EvaluationConfiguration):
        self.metrics_aggregator = MetricsAggregator()
        self.metrics_calculator = RetrievalMetricsCalculator()
        #configuration_id => research_method => query_id => list of chunks
        self.chunks_aggregator = {}  # Store chunks for potential further analysis

    def evaluate_retrieval_results(
            self,
            config: RAGConfiguration,
            retrieval_function: callable  # Function that takes (query, collection, search_config) -> chunks
    ):
        """
        Evaluate retrieval results for all queries in the dataset at both document and section levels.

        Args:
            config: RAG configuration
            retrieval_function: Function to retrieve chunks (injected dependency)
        """
        df_dataset = process_dataset(config.evaluation_configuration, config.preprocess_configuration)

        print(f"Starting dual-level evaluation for collection: {config.collection_name}")
        print(f"Loaded {len(df_dataset)} queries from dataset")
        print("\nEvaluation Strategy:")
        print("  📊 Retrieval: Always fetch 50 chunks for comprehensive ranking evaluation")
        print("  📏 Precision/Recall: Use configured top_k chunks for fairness")
        print("  📈 NDCG/MRR: Use all 50 chunks for better ranking assessment")
        print("\nEvaluation Levels:")
        print("  📄 Document Level: Chunk matches if filename is correct")
        print("  📑 Section Level: Chunk matches if chapter/section/subsection is exact")

        # Process each search configuration
        for search_config in config.search_configs:
            print(f"\nEvaluating search type: {search_config.search_type.name} (top_k={search_config.top_k})")

            # Process each query in the dataset
            for index, row in df_dataset.iterrows():
                query_id = f"query_{index}"
                question = row['question']
                ground_truth = row['ground_truth']

                # Use injected retrieval function (should return 50 chunks)
                chunks = retrieval_function(question, config.collection_name, search_config)

                configuration_id = config.configuration_id(search_config.search_type)

                print(f"[DEBUG] Generated configuration_id: {configuration_id[:12]}...")
                print(f"[DEBUG] Query ID: {query_id}")

                # Initialize nested dict if this config hasn't been seen
                if configuration_id not in self.chunks_aggregator:
                    self.chunks_aggregator[configuration_id] = {}
                    print(f"[DEBUG] Initialized new config entry for: {configuration_id[:12]}...")

                if query_id not in self.chunks_aggregator[configuration_id]:
                    self.chunks_aggregator[configuration_id][query_id] = []
                    print(f"[DEBUG] Initialized new query entry for: {query_id}")

                # Store chunks
                self.chunks_aggregator[configuration_id][query_id] = chunks


                # Evaluate the retrieved chunks at both levels
                metrics = self._evaluate_single_query(
                    query_id=query_id,
                    query_text=question,
                    chunks=chunks,
                    ground_truth=ground_truth,
                    search_config=search_config,
                    config=config
                )

                # Add to aggregator
                self.metrics_aggregator.add_metrics(metrics)

                if (index + 1) % 10 == 0:
                    print(f"Processed {index + 1}/{len(df_dataset)} queries for {search_config.search_type.name}")

        print(f"\nDual-level evaluation completed for {config.collection_name}")

    def _evaluate_single_query(
            self,
            query_id: str,
            query_text: str,
            chunks: List[dict],
            ground_truth: List[GroundTruthDocumentRecord],
            search_config: SearchConfiguration,
            config: RAGConfiguration
    ) -> QueryEvaluationMetrics:
        """
        Evaluate retrieved chunks against ground truth for a single query at both document and section levels.

        Uses different chunk sets for different metrics:
        - Precision/Recall: Uses top_k chunks from search_config
        - NDCG/MRR: Uses all 50 retrieved chunks for better ranking evaluation

        Returns:
            QueryEvaluationMetrics: Computed evaluation metrics for both levels
        """

        # Calculate dual-level precision, recall, and F1-score using top_k chunks
        (doc_precision, doc_recall, doc_f1), (sec_precision, sec_recall, sec_f1) = \
            self.metrics_calculator.calculate_dual_level_metrics(
                chunks, ground_truth, top_k=search_config.top_k
            )

        # Calculate dual-level NDCG using all 50 chunks for ranking evaluation
        doc_ndcg, sec_ndcg = self.metrics_calculator.calculate_dual_level_ndcg(
            chunks, ground_truth, k=50  # Always use 50 chunks for NDCG
        )

        # Calculate dual-level MRR using all 50 chunks
        doc_mrr, sec_mrr = self.metrics_calculator.calculate_dual_level_mrr(
            chunks, ground_truth
        )

        # NEW: Calculate dual-level MAP using all 50 chunks
        doc_map, sec_map = self.metrics_calculator.calculate_dual_level_map(
            chunks, ground_truth, k=50
        )

        # Generate configuration ID
        configuration_id = config.configuration_id(search_config.search_type)  # self.metrics_calculator.generate_configuration_id(search_config.search_type, config)

        # Count relevant retrieved chunks at both levels (using top_k for consistency with P/R)
        doc_relevant_count, sec_relevant_count = \
            self.metrics_calculator.count_relevant_chunks_dual_level(
                chunks, ground_truth, top_k=search_config.top_k
            )

        return QueryEvaluationMetrics(
            query_id=query_id,
            query_text=query_text,

            # Document level metrics (more permissive - filename match only)
            document_precision=doc_precision,
            document_recall=doc_recall,
            document_f1_score=doc_f1,
            document_ndcg=doc_ndcg,
            document_mrr=doc_mrr,
            document_map=doc_map,  # NEW


            # Section level metrics (strict - exact chapter/section/subsection match)
            section_precision=sec_precision,
            section_recall=sec_recall,
            section_f1_score=sec_f1,
            section_ndcg=sec_ndcg,
            section_mrr=sec_mrr,
            section_map=sec_map,  # NEW

            search_type=search_config.search_type,
            configuration_id=configuration_id,
            ground_truth_count=len(ground_truth),
            retrieved_count_total=len(chunks),  # Should be 50
            retrieved_count_evaluated=search_config.top_k,  # Used for P/R
            document_relevant_retrieved_count=doc_relevant_count,
            section_relevant_retrieved_count=sec_relevant_count
        )


    def __store_retrieved_chunks(self):
        """Store retrieved chunks to JSON files, one per configuration_id"""

        if not self.chunks_aggregator:
            print("No chunks to store")
            return

        # Get report directory
        report_dir = self.report_generator.report_dir
        chunks_dir = report_dir / "retrieved_chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Save one JSON file per configuration_id
        for config_id, query_chunks in self.chunks_aggregator.items():
            # Create readable filename from config_id (shortened hash)
            short_config_id = config_id[:12]
            filename = f"chunks_{short_config_id}.json"
            filepath = chunks_dir / filename

            # Prepare data structure
            output_data = {
                'configuration_id': config_id,
                'total_queries': len(query_chunks),
                'queries': {}
            }

            # Process each query and its chunks
            for query_id, chunks in query_chunks.items():
                query_data = {
                    'chunk_count': len(chunks),
                    'chunks': []
                }

                # Extract chunk information with metadata
                for idx, chunk in enumerate(chunks):
                    chunk_info = {
                        'rank': idx + 1,
                        'text': chunk.get('text', ''),
                        'metadata': {
                            'filename': chunk.get('filename'),
                            'chapter': chunk.get('chapter'),
                            'section': chunk.get('section'),
                            'subsection': chunk.get('subsection'),
                            'division': chunk.get('division')
                        }
                    }
                    query_data['chunks'].append(chunk_info)

                output_data['queries'][query_id] = query_data

            # Write to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Saved chunks for config {short_config_id}: {filepath}")




    def _create_configuration_mappings(self, configs: List[RAGConfiguration]) -> dict:
        """
        Create configuration mappings from RAGConfiguration objects.
        NOW INCLUDES PREPROCESSING CONFIGURATION.

        Args:
            configs: List of RAGConfiguration objects

        Returns:
            Dictionary mapping configuration_id to configuration details
        """
        config_mappings = {}

        for config in configs:
            for search_config in config.search_configs:
                # Generate the same configuration ID as used in metrics
                configuration_id = config.configuration_id(search_config.search_type) #self.metrics_calculator.generate_configuration_id(search_config.search_type, config)

                # Extract embedder model name
                embedder_name = config.model_manager.config.get('embedding_model_name', 'Unknown')

                # Extract search strategy name
                search_strategy = search_config.search_type.name.lower().replace('_', ' ')

                # Extract chunking strategy name
                chunking_strategy = getattr(config.chunking, 'name', 'Unknown')
                if hasattr(config.chunking, '_name'):
                    chunking_strategy = config.chunking._name

                # NEW: Extract preprocessing configuration
                preprocessing_info = "lowercase" if config.preprocess_configuration.lowercase else "no-lowercase"

                config_mappings[configuration_id] = {
                    'embedder': embedder_name,
                    'search_strategy': search_strategy,
                    'chunking_strategy': chunking_strategy,
                    'preprocessing': preprocessing_info  # NEW FIELD
                }

        return config_mappings

    def _print_evaluation_summary(self):
        """Print a comprehensive summary of dual-level evaluation results to console"""
        config_averages = self.metrics_aggregator.calculate_average_metrics_by_config()

        if not config_averages:
            print("No evaluation results to summarize")
            return

        print("\n" + "=" * 100)
        print("DUAL-LEVEL EVALUATION SUMMARY (Document Level vs Section Level)")
        print("=" * 100)

        print(f"Total queries evaluated: {len(self.metrics_aggregator.metrics)}")
        print(f"Total configurations tested: {len(config_averages)}")
        print("\nEvaluation Strategy:")
        print("  • Retrieved 50 chunks per query for comprehensive ranking evaluation")
        print("  • Used configured top_k for Precision/Recall calculations")
        print("  • Used all 50 chunks for NDCG/MRR calculations")
        print("  • Document-level: True positive if chunk filename matches ground truth")
        print("  • Section-level: True positive if exact chapter/section/subsection matches")
        print("  • Document metrics should be >= Section metrics (more permissive criteria)")

        print("\n📄 DOCUMENT LEVEL PERFORMANCE (Filename Match Only)")
        print("-" * 100)
        print(
            f"{'Config ID':<15} {'Queries':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'NDCG':<8} {'MRR':<8}")
        print("-" * 100)

        for config_id, metrics in config_averages.items():
            short_id = config_id[:12] + "..." if len(config_id) > 15 else config_id
            print(f"{short_id:<15} {metrics['query_count']:<8} "
                  f"{metrics['document_precision']:<10.4f} {metrics['document_recall']:<8.4f} "
                  f"{metrics['document_f1_score']:<10.4f} {metrics['document_ndcg']:<8.4f} "
                  f"{metrics['document_mrr']:<8.4f} {metrics['document_map']:<8.4f}")  # ← ADD


        print("\n📑 SECTION LEVEL PERFORMANCE (Exact Chapter/Section/Subsection Match)")
        print("-" * 100)
        print(
            f"{'Config ID':<15} {'Queries':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'NDCG':<8} {'MRR':<8}")
        print("-" * 100)

        for config_id, metrics in config_averages.items():
            short_id = config_id[:12] + "..." if len(config_id) > 15 else config_id
            print(f"{short_id:<15} {metrics['query_count']:<8} "
                  f"{metrics['section_precision']:<10.4f} {metrics['section_recall']:<8.4f} "
                  f"{metrics['section_f1_score']:<10.4f} {metrics['section_ndcg']:<8.4f} "
                  f"{metrics['section_mrr']:<8.4f}")

        print("\n🏆 BEST CONFIGURATIONS BY METRIC")
        print("\nDocument Level Champions:")
        for metric_name in ['document_precision', 'document_recall', 'document_f1_score', 'document_ndcg',
                            'document_mrr']:
            best_config, best_value = self.metrics_aggregator.get_best_configuration_by_metric(metric_name)
            if best_config:
                metric_display = metric_name.replace('document_', '').replace('_', ' ').title()
                short_config = best_config[:20] + "..." if len(best_config) > 23 else best_config
                print(f"  🥇 Best {metric_display}: {short_config} ({best_value:.4f})")

        print("\nSection Level Champions:")
        for metric_name in ['section_precision', 'section_recall', 'section_f1_score', 'section_ndcg', 'section_mrr']:
            best_config, best_value = self.metrics_aggregator.get_best_configuration_by_metric(metric_name)
            if best_config:
                metric_display = metric_name.replace('section_', '').replace('_', ' ').title()
                short_config = best_config[:20] + "..." if len(best_config) > 23 else best_config
                print(f"  🥇 Best {metric_display}: {short_config} ({best_value:.4f})")

        # Performance analysis
        print("\n📊 PERFORMANCE ANALYSIS")
        self._analyze_document_vs_section_performance(config_averages)

        print("=" * 100)

    def _analyze_document_vs_section_performance(self, config_averages: dict):
        """Analyze the relationship between document-level and section-level performance"""

        improvements = []
        issues = []

        for config_id, metrics in config_averages.items():
            doc_f1 = metrics['document_f1_score']
            sec_f1 = metrics['section_f1_score']
            improvement = doc_f1 - sec_f1

            if improvement > 0:
                improvements.append((config_id, improvement, doc_f1, sec_f1))
            elif improvement < 0:
                issues.append((config_id, improvement, doc_f1, sec_f1))

        if improvements:
            avg_improvement = sum(imp[1] for imp in improvements) / len(improvements)
            print(f"✅ Average Document-Section F1 improvement: {avg_improvement:.4f}")

            best_improvement = max(improvements, key=lambda x: x[1])
            short_config = best_improvement[0][:20] + "..." if len(best_improvement[0]) > 23 else best_improvement[0]
            print(f"🚀 Largest improvement: {short_config}")
            print(f"   Document F1: {best_improvement[2]:.4f}, Section F1: {best_improvement[3]:.4f}")
            print(f"   Improvement: +{best_improvement[1]:.4f}")

        if issues:
            print(f"\n⚠️  Warning: {len(issues)} configuration(s) show section > document performance")
            print("   This is unexpected and may indicate evaluation issues:")
            for config_id, diff, doc_f1, sec_f1 in issues:
                short_config = config_id[:20] + "..." if len(config_id) > 23 else config_id
                print(f"   • {short_config}: Doc={doc_f1:.4f}, Sec={sec_f1:.4f} (diff: {diff:.4f})")

    def clear_metrics(self):
        """Clear accumulated metrics (useful for processing multiple collections)"""
        self.metrics_aggregator.clear()

    def get_metrics_summary(self) -> dict:
        """Get a summary of current dual-level metrics for programmatic access"""
        return {
            'total_queries': len(self.metrics_aggregator.metrics),
            'configuration_averages': self.metrics_aggregator.calculate_average_metrics_by_config(),
            'best_configurations': {
                # Document level best
                'document_precision': self.metrics_aggregator.get_best_configuration_by_metric('document_precision'),
                'document_recall': self.metrics_aggregator.get_best_configuration_by_metric('document_recall'),
                'document_f1_score': self.metrics_aggregator.get_best_configuration_by_metric('document_f1_score'),
                'document_ndcg': self.metrics_aggregator.get_best_configuration_by_metric('document_ndcg'),
                'document_mrr': self.metrics_aggregator.get_best_configuration_by_metric('document_mrr'),
                # Section level best
                'section_precision': self.metrics_aggregator.get_best_configuration_by_metric('section_precision'),
                'section_recall': self.metrics_aggregator.get_best_configuration_by_metric('section_recall'),
                'section_f1_score': self.metrics_aggregator.get_best_configuration_by_metric('section_f1_score'),
                'section_ndcg': self.metrics_aggregator.get_best_configuration_by_metric('section_ndcg'),
                'section_mrr': self.metrics_aggregator.get_best_configuration_by_metric('section_mrr')
            }
        }

    def get_performance_insights(self) -> dict:
        """Get insights about document vs section level performance differences"""
        config_averages = self.metrics_aggregator.calculate_average_metrics_by_config()

        insights = {
            'average_improvements': {},
            'best_dual_performers': {},
            'problematic_configs': []
        }

        if not config_averages:
            return insights

        # Calculate average improvements
        metrics = ['precision', 'recall', 'f1_score', 'ndcg', 'mrr']
        for metric in metrics:
            doc_values = [config[f'document_{metric}'] for config in config_averages.values()]
            sec_values = [config[f'section_{metric}'] for config in config_averages.values()]

            avg_doc = sum(doc_values) / len(doc_values)
            avg_sec = sum(sec_values) / len(sec_values)
            insights['average_improvements'][metric] = avg_doc - avg_sec

        # Find best dual performers (good at both levels)
        for metric in metrics:
            best_combined = None
            best_score = 0

            for config_id, config_metrics in config_averages.items():
                # Combined score: average of document and section level
                combined_score = (config_metrics[f'document_{metric}'] + config_metrics[f'section_{metric}']) / 2
                if combined_score > best_score:
                    best_score = combined_score
                    best_combined = (config_id, combined_score,
                                     config_metrics[f'document_{metric}'],
                                     config_metrics[f'section_{metric}'])

            if best_combined:
                insights['best_dual_performers'][metric] = best_combined

        # Find problematic configs (section > document)
        for config_id, config_metrics in config_averages.items():
            issues = []
            for metric in metrics:
                if config_metrics[f'section_{metric}'] > config_metrics[f'document_{metric}']:
                    issues.append(metric)

            if issues:
                insights['problematic_configs'].append({
                    'config_id': config_id,
                    'problematic_metrics': issues,
                    'metrics': config_metrics
                })
        return insights