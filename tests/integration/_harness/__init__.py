"""Shared, backend-parametrized harness for real-server integration tests.

Defines :class:`MCPServerIntegrationTest`, which drives a real MCP server
subprocess (launched via ``tests/run_server.py``) through the FastMCP client
and asserts the full tool surface. The class is assembled from one mixin per
tool domain or concern, each built on ``HarnessCore`` in ``core.py``, which
owns the client, the server environment and the result list;
``run_all_tests`` here fixes the order the checks run in. The harness is
backend-agnostic: the same assertion methods run against SQLite or
PostgreSQL depending on the ``backend`` / ``pg_url`` passed to the
constructor; on PostgreSQL ``pg_admin_url`` names the superuser connection
the row counts no tool shows are read through, and ``rls`` the row-level
security credentials every server of the run starts with. The per-backend
pytest entry points live in ``tests/integration/sqlite/test_real_server.py``
and ``tests/integration/postgresql/test_real_server.py``.

No module in this package is named ``test_*``, so pytest does not collect it
directly; the per-backend entry-point modules import it.
"""

from tests.integration._harness.access_control import AccessControlMixin
from tests.integration._harness.access_scoping import AccessScopingMixin
from tests.integration._harness.backend_runtime import BackendRuntimeMixin
from tests.integration._harness.batch import BatchMixin
from tests.integration._harness.batch_atomicity import BatchAtomicityMixin
from tests.integration._harness.batch_conformance import BatchConformanceMixin
from tests.integration._harness.chunking import ChunkingMixin
from tests.integration._harness.delete import DeleteMixin
from tests.integration._harness.discovery import DiscoveryMixin
from tests.integration._harness.metadata_filters import MetadataFiltersMixin
from tests.integration._harness.metadata_filters_indexed import MetadataFiltersIndexedMixin
from tests.integration._harness.metadata_filters_membership import MetadataFiltersMembershipMixin
from tests.integration._harness.metadata_filters_numeric import MetadataFiltersNumericMixin
from tests.integration._harness.metadata_patch import MetadataPatchMixin
from tests.integration._harness.middleware import MiddlewareMixin
from tests.integration._harness.navigation import NavigationMixin
from tests.integration._harness.reranking import RerankingMixin
from tests.integration._harness.retrieve import RetrieveMixin
from tests.integration._harness.search_browse import SearchBrowseMixin
from tests.integration._harness.search_cross_tool import SearchCrossToolMixin
from tests.integration._harness.search_explain import SearchExplainMixin
from tests.integration._harness.search_fts import SearchFtsMixin
from tests.integration._harness.search_fts_boolean import SearchFtsBooleanMixin
from tests.integration._harness.search_fts_filters import SearchFtsFiltersMixin
from tests.integration._harness.search_hybrid import SearchHybridMixin
from tests.integration._harness.search_hybrid_filters import SearchHybridFiltersMixin
from tests.integration._harness.search_ranking import SearchRankingMixin
from tests.integration._harness.search_semantic import SearchSemanticMixin
from tests.integration._harness.search_validation import SearchValidationMixin
from tests.integration._harness.server import ServerMixin
from tests.integration._harness.sharing import SharingMixin
from tests.integration._harness.store import StoreMixin
from tests.integration._harness.store_dedup import StoreDedupMixin
from tests.integration._harness.summary import SummaryMixin
from tests.integration._harness.sync import SyncMixin
from tests.integration._harness.tags_images import TagsImagesMixin
from tests.integration._harness.update import UpdateMixin

# The pytest-timeout ceiling of one full harness run. One run drives every check
# through one server and takes five to ten minutes, depending on the load of the
# machine, so the suite-wide default of 600 seconds cuts off runs that would pass; the
# entry points carry this ceiling instead, which still stops a hang.
REAL_SERVER_TIMEOUT_S = 1200


class MCPServerIntegrationTest(
    StoreMixin,
    StoreDedupMixin,
    RetrieveMixin,
    UpdateMixin,
    MetadataPatchMixin,
    DeleteMixin,
    AccessControlMixin,
    AccessScopingMixin,
    TagsImagesMixin,
    BatchMixin,
    BatchAtomicityMixin,
    BatchConformanceMixin,
    DiscoveryMixin,
    SummaryMixin,
    MiddlewareMixin,
    ServerMixin,
    BackendRuntimeMixin,
    MetadataFiltersMixin,
    MetadataFiltersNumericMixin,
    MetadataFiltersMembershipMixin,
    MetadataFiltersIndexedMixin,
    SearchBrowseMixin,
    SearchExplainMixin,
    SearchValidationMixin,
    SearchSemanticMixin,
    SearchFtsMixin,
    SearchFtsFiltersMixin,
    SearchFtsBooleanMixin,
    SearchHybridMixin,
    SearchHybridFiltersMixin,
    SearchCrossToolMixin,
    SearchRankingMixin,
    ChunkingMixin,
    RerankingMixin,
    NavigationMixin,
    SyncMixin,
    SharingMixin,
):
    """Integration test for real MCP Context Storage Server."""

    async def run_all_tests(self) -> bool:
        """Run all tests and report results.

        Returns:
            bool: True if all tests passed.
        """
        print('\n' + '=' * 50)
        print(f'MCP SERVER INTEGRATION TEST ({self.backend}, {self.client_mode} client mode)')
        print('=' * 50)

        # Start server
        if not await self.start_server():
            print('[ERROR] Failed to start server')
            await self.cleanup()
            return False

        # Connect client
        if not await self.connect_client():
            print('[ERROR] Failed to connect client')
            await self.cleanup()
            return False

        # Run all tests
        tests = [
            ('Store Context', self.test_store_context),
            ('Search Context', self.test_search_context),
            ('Search Context Date Filtering', self.test_search_context_with_date_filtering),
            ('Metadata Filtering', self.test_metadata_filtering),
            ('Metadata Filter Nested Path', self.test_metadata_filter_nested_path),
            ('Metadata Filter LIKE Wildcard Literal', self.test_metadata_filter_like_wildcard_literal),
            ('Metadata Filter Numeric Type Parity', self.test_metadata_filter_numeric_type_parity),
            ('Metadata Filter Float Precision Parity', self.test_metadata_filter_float_precision_parity),
            (
                'Metadata Filter High-Magnitude Int Float-Param Parity',
                self.test_metadata_filter_high_magnitude_int_float_param_parity,
            ),
            (
                'Metadata Filter High-Magnitude Float Roundtrip Parity',
                self.test_metadata_filter_high_magnitude_float_roundtrip_parity,
            ),
            (
                'Metadata Filter Out-of-Range Numeric Parity',
                self.test_metadata_filter_out_of_range_numeric_parity,
            ),
            (
                'Array Contains High-Magnitude Float Parity',
                self.test_array_contains_high_magnitude_float_parity,
            ),
            ('Metadata Filter Boolean Type Parity', self.test_metadata_filter_boolean_type_parity),
            ('Metadata Filter GLOB Special Literal', self.test_metadata_filter_glob_special_literal),
            ('Semantic Hybrid Metadata Dict', self.test_semantic_hybrid_metadata_is_dict),
            ('Array Contains Operator', self.test_array_contains_operator),
            ('Array Contains Non-Array Field', self.test_array_contains_non_array_field),
            ('Get Context by IDs', self.test_get_context_by_ids),
            ('Delete Context', self.test_delete_context),
            ('Update Context', self.test_update_context),
            ('Visibility Lifecycle', self.test_visibility_lifecycle),
            ('Metadata Patch Deep Merge', self.test_metadata_patch_deep_merge),
            ('Metadata Patch RFC 7396 Full Compliance', self.test_metadata_patch_rfc7396_full_compliance),
            ('Metadata Patch Successive Patches', self.test_metadata_patch_successive_patches),
            ('Metadata Patch Type Conversions', self.test_metadata_patch_type_conversions),
            ('List Threads', self.test_list_threads),
            ('Get Statistics', self.test_get_statistics),
            ('Store Context Batch', self.test_store_context_batch),
            ('Update Context Batch', self.test_update_context_batch),
            ('Update Context Batch Version Guard', self.test_update_context_batch_version_guard),
            ('Delete Context Batch', self.test_delete_context_batch),
            ('Semantic Search', self.test_semantic_search_context),
            ('Semantic Search Date Filtering', self.test_semantic_search_context_with_date_filtering),
            ('Semantic Search Metadata Filtering', self.test_semantic_search_context_with_metadata_filters),
            ('Search Context Invalid Filter Error', self.test_search_context_invalid_filter_returns_error),
            ('Search Context Blank Tags Error', self.test_search_context_blank_tags_returns_error),
            (
                'Search Context Oversized Metadata Filters',
                self.test_search_context_oversized_metadata_filters_rejected,
            ),
            ('NUL Input Does Not Trip Breaker', self.test_nul_input_does_not_trip_breaker),
            ('Semantic Search Invalid Filter Error', self.test_semantic_search_invalid_filter_returns_error),
            ('FTS Search', self.test_fts_search_context),
            ('FTS Search Invalid Filter Error', self.test_fts_search_invalid_filter_returns_error),
            ('FTS Boolean Mode', self.test_fts_boolean_mode),
            ('FTS Date Range Filter', self.test_fts_date_range_filter),
            ('FTS Metadata Filter', self.test_fts_metadata_filter),
            ('FTS Metadata Filter Key Substring', self.test_fts_metadata_filter_key_substring),
            ('FTS Advanced Metadata Filters', self.test_fts_advanced_metadata_filters),
            ('FTS Pagination Offset', self.test_fts_pagination_offset),
            ('FTS Highlight Snippets', self.test_fts_highlight_snippets),
            ('Hybrid Search', self.test_hybrid_search_context),
            ('Hybrid Search Adaptive FTS Mode', self.test_hybrid_search_adaptive_fts_mode),
            ('Search Tools Content Type Filter', self.test_search_tools_content_type_filter),
            ('Search Tools Include Images', self.test_search_tools_include_images),
            ('Search Tools Tags Filter', self.test_search_tools_tags_filter),
            ('Semantic Search Offset Pagination', self.test_semantic_search_offset_pagination),
            ('Hybrid Search Metadata Filtering', self.test_hybrid_search_metadata_filtering),
            ('Hybrid Search Date Range Filtering', self.test_hybrid_search_date_range_filtering),
            ('Hybrid Search Offset Pagination', self.test_hybrid_search_offset_pagination),
            ('Explain Query Statistics', self.test_explain_query_statistics),
            # Chunking and Reranking Tests
            ('Statistics Chunking Reranking Info', self.test_statistics_chunking_reranking_info),
            ('Statistics Summary Info', self.test_statistics_summary_info),
            ('Chunking Creates Multiple Embeddings', self.test_chunking_creates_multiple_embeddings),
            ('Chunking Long Document Storage', self.test_chunking_long_document_storage),
            ('Reranking Adds Score to Results', self.test_reranking_adds_score_to_results),
            ('Reranking in FTS Search', self.test_reranking_in_fts_search),
            ('Reranking in Hybrid Search', self.test_reranking_in_hybrid_search),
            ('Chunking Deduplication in Search', self.test_chunking_deduplication_in_search),
            ('Chunking Disabled Single Embedding', self.test_chunking_disabled_single_embedding),
            ('Reranking Disabled No Score', self.test_reranking_disabled_no_score),
            ('Chunking Reranking Integration', self.test_chunking_reranking_integration),
            ('Overfetch Chain Verification', self.test_overfetch_chain_verification),
            # Search Limit Tests
            ('Search Context Limit Clamping', self.test_search_context_limit_clamping),
            # Protocol Era And Server Identity Tests
            ('Client Negotiates Requested Protocol Era', self.test_client_negotiates_requested_protocol_era),
            ('Server Version Is Project Version', self.test_server_version_is_project_version),
            # Deduplication Data Integrity Tests
            ('Dedup Data Integrity', self.test_store_context_deduplication_data_integrity),
            ('Store Deduplicate False Inserts Fresh', self.test_store_deduplicate_false_inserts_fresh),
            # Edge Case Tests
            ('Store Context Empty Text', self.test_store_context_empty_text),
            ('Store Context Max Size Image', self.test_store_context_max_size_image),
            ('Search Context No Results', self.test_search_context_no_results),
            ('Delete Context Nonexistent ID', self.test_delete_context_nonexistent_id),
            ('Update Context Nonexistent ID', self.test_update_context_nonexistent_id),
            ('Get Context By IDs Partial Match', self.test_get_context_by_ids_partial_match),
            ('List Threads With Filter', self.test_list_threads_empty_database),
            ('Batch Operations Atomic Rollback', self.test_batch_operations_atomic_rollback),
            ('Batch Operations Non-Atomic Partial', self.test_batch_operations_non_atomic_partial),
            # Summary Field Tests
            ('Get Context By IDs Omits Summary By Default', self.test_get_context_by_ids_omits_summary_by_default),
            ('Get Context By IDs Includes Summary When Enabled', self.test_get_context_by_ids_includes_summary_when_enabled),
            ('Search Context Summary Display', self.test_search_context_summary_display),
            ('Batch Store Summary Field', self.test_batch_store_summary_field),
            # Generation-First Pattern Tests
            ('Store Context Generation First', self.test_store_context_generation_first_return_exceptions),
            ('Store Context Long Document In Batches', self.test_store_context_long_document_in_batches),
            ('Batch Store Generation First', self.test_batch_store_generation_first_return_exceptions),
            # Middleware JSON String Deserializer Tests
            ('Middleware Deserializes Stringified Tags', self.test_middleware_deserializes_stringified_tags),
            ('Middleware Deserializes Stringified Metadata', self.test_middleware_deserializes_stringified_metadata),
            ('Middleware Preserves String Text', self.test_middleware_preserves_string_text),
            ('Middleware Deserializes Stringified Context IDs', self.test_middleware_deserializes_stringified_context_ids),
            # Summary Env Var Tests
            ('Summary Env Vars Accepted', self.test_summary_env_vars_accepted),
            # Discovery, Write-Path, and Search-Mode Tests
            ('List Threads Populated Database', self.test_list_threads_with_populated_database),
            ('List Threads Pagination', self.test_list_threads_pagination),
            ('Session Pooler Validation No-op on SQLite', self.test_session_pooler_validation_noop_on_sqlite),
            ('Update Triggers Embedding Regen', self.test_update_context_triggers_embedding_regeneration),
            ('Batch Store Dedup Within Batch', self.test_store_context_batch_dedup_within_batch),
            ('Batch Store Deduplicate False Per Entry', self.test_store_batch_deduplicate_false_per_entry),
            ('Hybrid Search Graceful Degradation', self.test_hybrid_search_graceful_degradation_fts_only),
            ('Content Type Auto Detection', self.test_content_type_auto_detection_multimodal),
            ('Update Image Without MIME Type', self.test_update_context_image_without_mime_type_integration),
            ('Batch Update Preserves Content Type', self.test_update_context_batch_content_type_correction),
            ('Content Type Filter Multimodal', self.test_search_context_content_type_filter_multimodal),
            ('FTS Match Stemming', self.test_fts_search_match_mode_stemming),
            ('FTS NOT Operator', self.test_fts_search_not_operator_exclusion),
            ('FTS Malformed Boolean Parity', self.test_fts_search_malformed_boolean_parity),
            ('Hybrid RRF Score Ordering', self.test_hybrid_search_rrf_scores_ordering),
            ('Explain Query False No Stats', self.test_search_context_explain_query_false_no_stats),
            ('Dedup Preserves Tags', self.test_store_context_dedup_preserves_tags_when_none),
            ('Dedup Interleaving Check', self.test_store_context_dedup_interleaving_check),
            ('Batch Non-Atomic Partial', self.test_update_context_batch_non_atomic_generation_failure),
            ('Health Endpoint', self.test_health_endpoint_returns_ok),
            # Batch/Non-Batch Conformance Tests
            ('Store Batch Conformance', self.test_store_batch_single_matches_store_nonbatch),
            (
                'Store Batch Conformance Deduplicate False',
                self.test_store_batch_single_matches_store_nonbatch_deduplicate_false,
            ),
            ('Update Batch Conformance', self.test_update_batch_single_matches_update_nonbatch),
            ('Delete Batch Conformance', self.test_delete_batch_single_matches_delete_nonbatch),
            # Tool-Surface Coverage (run on both backends)
            ('Metadata Filter Operators Comprehensive', self.test_metadata_filter_operators_comprehensive),
            ('Metadata NOT_IN Numeric Over Non-Number Row Parity',
             self.test_metadata_not_in_numeric_over_nonnumber_row_parity),
            ('Image Attachment Cascade Delete', self.test_image_attachment_cascade_delete),
            ('Tags Lowercase Normalization', self.test_tags_lowercase_normalization),
            ('Tool Annotations Exposed To Client', self.test_tool_annotations_exposed_to_client),
            ('Search Context Offset Pagination', self.test_search_context_offset_pagination),
            ('Grep Context Literal Regex Unicode', self.test_grep_context_literal_regex_unicode),
            ('Read Context Range Clamp And Composition', self.test_read_context_range_clamp_and_composition),
            ('Navigate Context Outline And Node Read', self.test_navigate_context_outline_and_node_read),
            ('Index Tree Node Summaries And Statistics', self.test_index_tree_node_summaries_and_statistics),
            ('Prefix Id Resolution Returns Canonical Id', self.test_prefix_id_resolution_returns_canonical_id),
            ('Grep Keyset Scan Exhaustive Beyond Limit', self.test_grep_keyset_scan_exhaustive_beyond_limit),
            ('Grep Scan Cap Boundary Truncation', self.test_grep_scan_cap_boundary_truncation),
            ('Force-Off Removes Search Tool', self.test_force_off_removes_search_tool),
            # Cross-backend parity checks: out-of-int64 simple metadata filter,
            # FTS validation-before-empty-query-short-circuit, and dedup content_type
            # plus image preservation on an image-less retransmit.
            ('Simple Metadata Out-Of-Int64 Rejected', self.test_search_simple_metadata_out_of_int64_rejected),
            ('FTS Empty-Query Invalid Filter Raises', self.test_fts_all_stopword_query_with_invalid_filter_returns_error),
            ('Dedup Preserves Content Type And Image', self.test_store_context_dedup_preserves_content_type_and_image),
            # Cross-backend parity checks: the shared connection-metrics contract,
            # write-queue delivery across idle windows, null-named metadata path segments,
            # multi-member numeric IN/NOT_IN, tied-score pagination, embedded-quote FTS
            # terms, the filters_applied tally, tag write caps, updated_at stamping,
            # per-image metadata typing, grep request clamping, and embedding cleanup.
            ('Connection Metrics Cross-Backend Contract', self.test_connection_metrics_cross_backend_contract),
            ('Write Queue Idle Windows And Bursts', self.test_write_queue_survives_idle_windows_and_bursts),
            ('Metadata Filter Null Path Segment Parity', self.test_metadata_filter_null_path_segment_parity),
            (
                'Metadata Filter Numeric IN Mixed Members Parity',
                self.test_metadata_filter_numeric_in_mixed_members_parity,
            ),
            ('Tied-Score Pagination Parity', self.test_tied_score_pagination_parity),
            ('FTS Embedded Quote Term Parity', self.test_fts_embedded_quote_term_parity),
            ('Filters Applied Agreement Across Search Tools', self.test_filters_applied_agreement_across_search_tools),
            ('Tag Write Caps Parity', self.test_tag_write_caps_parity),
            ('Update Advances updated_at For Every Variant', self.test_update_context_advances_updated_at),
            ('Image Metadata JSON String Contract', self.test_image_metadata_json_string_contract),
            ('Grep Context Request Caps Clamped', self.test_grep_context_request_caps_clamped),
            ('Delete Removes Embedding Rows', self.test_delete_removes_embedding_rows),
            # Cross-backend parity checks: the completed-operation counter for
            # transactional writes, the idle writer recycle, fixed-depth ranked pagination
            # and its depth hint, mutually exclusive delete selectors, typed indexed
            # metadata casts, byte-wise text ordering, tag deduplication, per-image
            # metadata fidelity, literal markup in a ranked document, and a deeply nested
            # boolean query that must not charge the circuit breaker.
            ('Transactional Write Moves total_queries', self.test_transactional_write_moves_total_queries),
            ('Idle Writer Recycle Invisible To Callers', self.test_idle_writer_recycle_is_invisible_to_callers),
            ('Ranked Pagination Union Matches Single Page', self.test_ranked_pagination_union_matches_single_page),
            ('Ranked Depth Limit Hint', self.test_ranked_depth_limit_hint),
            ('Delete Context Rejects Both Selectors', self.test_delete_context_rejects_both_selectors),
            ('Indexed Metadata Container Length Parity', self.test_indexed_metadata_container_length_parity),
            ('Typed Indexed Metadata Cast Parity', self.test_typed_indexed_metadata_cast_parity),
            ('Collation Ordering Parity', self.test_collation_ordering_parity),
            ('Tag Deduplication Across Write Paths', self.test_tag_deduplication_across_write_paths),
            ('Image Metadata Empty String Preserved', self.test_image_metadata_empty_string_preserved),
            ('Literal Markup Survives Ranked Search', self.test_literal_markup_survives_ranked_search),
            ('FTS Deeply Nested Boolean Query Degrades', self.test_fts_deeply_nested_boolean_query_degrades),
            # Corpus sync: registration (ungated), plans that converge and stay inside
            # their ranges, the paged ingest state, manifest validation at and past the
            # cap, and the reserved ingest block guarded on every write tool.
            ('Sync Tools Registered', self.test_sync_tools_registered),
            ('Plan Context Sync Converges', self.test_plan_context_sync_converges),
            ('Plan Context Sync Ranges', self.test_plan_context_sync_ranges),
            ('List Ingest State Paged', self.test_list_ingest_state_paged),
            ('Plan Context Sync Rejects Invalid Manifest', self.test_plan_context_sync_rejects_invalid_manifest),
            ('Plan Context Sync Manifest At Cap', self.test_plan_context_sync_manifest_at_cap),
            ('Ingest Block Rejected', self.test_ingest_block_rejected),
            # Sharing: registration (ungated), idempotent shares, exact-row unshares, grants
            # outside the version check, argument refusals, the visibility and ownership
            # every read reports to the owner, and grants deleted with their entry.
            ('Sharing Tools Registered', self.test_sharing_tools_registered),
            ('Share Context Idempotent', self.test_share_context_idempotent),
            ('Unshare Context Exact Row', self.test_unshare_context_exact_row),
            ('Grants Leave Version And Updated At', self.test_grants_leave_version_and_updated_at),
            ('Sharing Rejects Invalid Arguments', self.test_sharing_rejects_invalid_arguments),
            ('Read-Back On Own Entries', self.test_read_back_on_own_entries),
            ('Delete Removes Grants', self.test_delete_removes_grants),
            # A second principal served from the same database reaches only the entries
            # the access model lets it read, and changes only those it may modify.
            ('Access Scoping Second Principal', self.test_access_scoping_second_principal),
        ]

        print('\nRunning tests...\n')

        for test_name, test_func in tests:
            print(f'Testing: {test_name}...')
            try:
                success = await test_func()
                if success:
                    print(f'  [OK] {test_name} passed')
                else:
                    print(f'  [FAIL] {test_name} failed')
            except Exception as e:
                print(f'  [ERROR] {test_name} error: {e}')
                self.test_results.append((test_name, False, f'Exception: {e}'))

        # Display results
        print('\n' + '=' * 50)
        print('TEST RESULTS')
        print('=' * 50)

        passed = 0
        failed = 0

        for test_name, result, details in self.test_results:
            status = '[PASS]' if result else '[FAIL]'
            print(f'{status}: {test_name}')
            if details:
                print(f'   Details: {details}')
            if result:
                passed += 1
            else:
                failed += 1

        total = passed + failed
        print(f'\nTotal: {passed}/{total} tests passed')

        # Cleanup
        await self.cleanup()

        return failed == 0
