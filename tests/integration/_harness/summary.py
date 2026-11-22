"""Real-server checks for summary display and configuration.

A search returns a long text truncated and marked so, get_context_by_ids omits
the summary key by default after a batch store and a batch update, and the
provider-specific summary environment variables load without error. The checks
hold with a summary provider configured and without one.
"""

from tests.integration._harness.core import HarnessCore


class SummaryMixin(HarnessCore):
    """Checks for summary display and configuration."""

    async def test_search_context_summary_display(self) -> bool:
        """Test that search_context returns a long text truncated and a short text whole.

        A text longer than the search preview comes back shortened and marked
        is_text_content_truncated, whether or not a summary provider wrote a
        summary for it; a short text comes back whole and unmarked.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_summary_display'
        assert self.client is not None  # Type guard for Pyright
        try:
            summary_display_thread = f'{self.test_thread_id}_summary_display'

            # Store a text longer than the preview a search returns. It is prose without
            # a repeated sentence: a summary model handed one sentence many times over
            # repeats it until its token limit.
            long_text = (
                'A small support team starts its day with the tickets that arrived overnight. '
                'The urgent ones are answered first, and each answer names the person who owns the next step. '
                'Before noon the team reads the open questions of the week and decides which of them '
                'need a fix in the product and which need a clearer page in the guide. '
                'The afternoon belongs to the fixes that take more than one person: two engineers pair on the change, '
                'a third reads it, and the ticket is closed only after the customer confirms the result. '
                'On Friday a short report lists what was closed, what is still open and what the team learned, '
                'so that the next week starts from a written state and not from memory.'
            )
            store_long = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': summary_display_thread,
                    'source': 'agent',
                    'text': long_text,
                },
            )
            long_data = self._extract_content(store_long)
            if not long_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Failed to store long context: {long_data}'),
                )
                return False

            # Store a short text entry (should not be truncated)
            short_text = 'Brief note'
            store_short = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': summary_display_thread,
                    'source': 'user',
                    'text': short_text,
                },
            )
            short_data = self._extract_content(store_short)
            if not short_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Failed to store short context: {short_data}'),
                )
                return False

            # Search for both entries
            search_result = await self.client.call_tool(
                'search_context',
                {
                    'thread_id': summary_display_thread,
                    'limit': 10,
                },
            )
            search_data = self._extract_content(search_result)
            if not search_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Search failed: {search_data}'),
                )
                return False

            results = search_data.get('results', [])
            if len(results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 results, got {len(results)}'),
                )
                return False

            # Find the long and short entries in results
            long_entry = None
            short_entry = None
            for r in results:
                if r.get('id') == long_data['context_id']:
                    long_entry = r
                elif r.get('id') == short_data['context_id']:
                    short_entry = r

            if long_entry is None or short_entry is None:
                self.test_results.append(
                    (test_name, False, 'Could not find both entries in search results'),
                )
                return False

            # Long text is truncated in search results
            if not long_entry.get('is_text_content_truncated', False):
                self.test_results.append(
                    (test_name, False,
                     'Long text should be truncated in search results'),
                )
                return False

            # Long entry's text_content should be shorter than original
            if len(long_entry['text_content']) >= len(long_text):
                self.test_results.append(
                    (test_name, False,
                     (f'Truncated text ({len(long_entry["text_content"])}) should be '
                      f'shorter than original ({len(long_text)})')),
                )
                return False

            # Short text should NOT be truncated
            if short_entry.get('is_text_content_truncated', True):
                self.test_results.append(
                    (test_name, False,
                     'Short text should not be truncated in search results'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 'Search correctly truncates long text, preserves short text, uses is_text_content_truncated'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_batch_store_summary_field(self) -> bool:
        """Test that get_context_by_ids omits the summary key of batch-written entries by default.

        The key is absent after store_context_batch and still absent after
        update_context_batch.

        Returns:
            bool: True if test passed.
        """
        test_name = 'batch_store_summary_field'
        assert self.client is not None  # Type guard for Pyright
        try:
            batch_summary_thread = f'{self.test_thread_id}_batch_summary'

            entries = [
                {
                    'thread_id': batch_summary_thread,
                    'source': 'user',
                    'text': 'Batch summary test entry one',
                },
                {
                    'thread_id': batch_summary_thread,
                    'source': 'agent',
                    'text': 'Batch summary test entry two',
                },
            ]

            result = await self.client.call_tool(
                'store_context_batch',
                {'entries': entries, 'atomic': True},
            )
            result_data = self._extract_content(result)

            if not result_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Batch store failed: {result_data}'),
                )
                return False

            if result_data.get('succeeded') != 2:
                self.test_results.append(
                    (test_name, False,
                     f'Expected 2 succeeded, got {result_data.get("succeeded")}'),
                )
                return False

            # Retrieve the stored entries and verify summary field
            context_ids = [r['context_id'] for r in result_data.get('results', [])]
            if len(context_ids) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 context_ids, got {len(context_ids)}'),
                )
                return False

            get_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': context_ids},
            )
            get_data = self._extract_content(get_result)
            results = get_data.get('results', [])

            if len(results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 entries from get, got {len(results)}'),
                )
                return False

            # Default config should OMIT summary from get_context_by_ids response
            for entry in results:
                if 'summary' in entry:
                    self.test_results.append(
                        (test_name, False,
                         ('summary field unexpectedly present in batch-stored entry '
                          '(default config should omit it)')),
                    )
                    return False

            # Test update_context_batch also omits summary field
            updates = [
                {'context_id': context_ids[0], 'text': 'Updated batch text one'},
            ]
            update_result = await self.client.call_tool(
                'update_context_batch',
                {'updates': updates, 'atomic': True},
            )
            update_data = self._extract_content(update_result)

            if not update_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Batch update failed: {update_data}'),
                )
                return False

            # Re-retrieve and verify summary is still omitted after update
            get_updated = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_ids[0]]},
            )
            updated_data = self._extract_content(get_updated)
            updated_results = updated_data.get('results', [])

            if len(updated_results) != 1:
                self.test_results.append(
                    (test_name, False,
                     f'Expected 1 updated entry, got {len(updated_results)}'),
                )
                return False

            if 'summary' in updated_results[0]:
                self.test_results.append(
                    (test_name, False,
                     ('summary field unexpectedly present after batch update '
                      '(default config should omit it)')),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 ('Batch store and update correctly omit summary field from '
                  'get_context_by_ids response by default')),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_summary_env_vars_accepted(self) -> bool:
        """Test that the provider-specific summary env vars are accepted by the running server.

        Verifies SUMMARY_OPENAI_REASONING_EFFORT and SUMMARY_ANTHROPIC_EFFORT
        settings are loaded without error by checking the server responds
        normally to get_statistics.

        Returns:
            bool: True if test passed.
        """
        test_name = 'summary_env_vars_accepted'
        assert self.client is not None  # Type guard for Pyright
        try:
            # The server subprocess inherits env vars set before Client creation.
            # SUMMARY_OPENAI_REASONING_EFFORT and SUMMARY_ANTHROPIC_EFFORT are
            # settings that the server reads at startup via SummarySettings.
            # If these env vars were invalid, the server would fail to start
            # (Pydantic validation error). The fact that we can call get_statistics
            # proves the server accepted these settings.

            # Verify server is responsive (settings loaded without error)
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            # Verify statistics response has expected structure
            if 'summary' not in stats_data:
                self.test_results.append(
                    (test_name, False, 'Missing summary section in statistics'),
                )
                return False

            summary_info = stats_data['summary']

            # Verify summary section has the enabled field
            if 'enabled' not in summary_info:
                self.test_results.append(
                    (test_name, False, 'Missing enabled field in summary statistics'),
                )
                return False

            # An active summary configuration reports its model, which confirms
            # SummarySettings loaded correctly
            is_summary_active = summary_info.get('enabled') and summary_info.get('available')
            if is_summary_active and 'model' not in summary_info:
                self.test_results.append(
                    (test_name, False, 'Missing model field in active summary'),
                )
                return False

            msg = (
                'Server accepted summary env vars '
                '(SUMMARY_OPENAI_REASONING_EFFORT, SUMMARY_ANTHROPIC_EFFORT)'
            )
            self.test_results.append((test_name, True, msg))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
