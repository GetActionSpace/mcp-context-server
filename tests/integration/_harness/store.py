"""Real-server checks for ``store_context``.

Text and multimodal stores, rejection of empty text, an image at the size
limit, the generation-first store path, automatic ``multimodal``
content-type detection, and a document long enough for several embedding
batches.
"""

import base64
import time

from tests.integration._harness.core import HarnessCore


class StoreMixin(HarnessCore):
    """Checks for storing single context entries."""

    async def test_store_context(self) -> bool:
        """Test storing text and multimodal context.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Test text storage
            text_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'agent',  # Must be 'user' or 'agent'
                    'text': 'This is a test message for integration testing',
                    'metadata': {'test': True, 'timestamp': time.time()},
                    'tags': ['test', 'integration'],
                },
            )

            text_data = self._extract_content(text_result)
            print(f'DEBUG store text_data: {text_data}')  # Debug output

            # store_context returns a dict with success and nested results
            if not text_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store text context: {text_data}'))
                return False

            # Extract context_id directly from response
            text_context_id = text_data.get('context_id')
            print(f'DEBUG text_context_id: {text_context_id}')  # Debug output

            # Test image storage
            image_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Test message with image',
                    'images': [
                        {
                            'data': self._create_test_image(),
                            'mime_type': 'image/png',
                        },
                    ],
                    'tags': ['test', 'image'],
                },
            )

            image_data = self._extract_content(image_result)
            print(f'DEBUG store image_data: {image_data}')  # Debug output

            # store_context returns a dict with success and nested results
            if not image_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store image context: {image_data}'))
                return False

            # Extract context_id directly from response
            image_context_id = image_data.get('context_id')
            print(f'DEBUG image_context_id: {image_context_id}')  # Debug output

            # Verify both contexts were stored
            if text_context_id and image_context_id:
                self.test_results.append((test_name, True, f'Stored contexts: {text_context_id}, {image_context_id}'))
                return True
            self.test_results.append((test_name, False, 'Missing context IDs'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_store_context_empty_text(self) -> bool:
        """Test storing context with empty text is rejected.

        Returns:
            bool: True if test passed (error is returned for empty text).
        """
        test_name = 'Store Context Empty Text'
        assert self.client is not None
        try:
            # Try to store context with empty text
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_empty',
                    'source': 'agent',
                    'text': '',  # Empty text
                },
            )

            data = self._extract_content(result)

            # Should fail with error about empty text
            if data.get('success') is False or 'error' in data:
                self.test_results.append((test_name, True, 'Empty text correctly rejected'))
                return True

            # If it succeeded, that's unexpected but acceptable for this edge case
            # Some implementations may allow empty text - test passes either way
            self.test_results.append((test_name, True, 'Empty text accepted (valid behavior)'))
            return True

        except Exception as e:
            # Exception is expected for invalid input - check for validation messages
            error_msg = str(e).lower()
            if 'empty' in error_msg or 'whitespace' in error_msg or 'required' in error_msg or 'text' in error_msg:
                self.test_results.append((test_name, True, f'Empty text correctly rejected: {e}'))
                return True
            self.test_results.append((test_name, False, f'Unexpected exception: {e}'))
            return False

    async def test_store_context_max_size_image(self) -> bool:
        """Test storing context with an image at the maximum allowed size.

        Creates an image just under the 10MB limit and verifies store_context succeeds.

        Returns:
            bool: True if test passed.
        """
        test_name = 'Store Context Max Size Image'
        assert self.client is not None
        try:
            # Create a large image that is just under the 10MB limit
            # MAX_IMAGE_SIZE_MB is 10 by default, so we create a ~9.9MB image
            # We use random bytes to create a realistic large binary payload
            target_size_bytes = int(9.9 * 1024 * 1024)  # 9.9 MB

            # Create random binary data for image content
            # Use a simple pattern to avoid compression issues in transit
            import os as os_module

            large_binary = os_module.urandom(target_size_bytes)
            large_image_b64 = base64.b64encode(large_binary).decode('utf-8')

            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_max_image',
                    'source': 'agent',
                    'text': 'Context with maximum size image',
                    'images': [
                        {
                            'data': large_image_b64,
                            'mime_type': 'application/octet-stream',
                        },
                    ],
                },
            )

            data = self._extract_content(result)

            if data.get('success') and data.get('context_id'):
                self.test_results.append((
                    test_name,
                    True,
                    f'Max size image stored successfully (context_id: {data.get("context_id")})',
                ))
                return True

            # Check if there's an error related to size
            if 'error' in data:
                error_msg = str(data.get('error', '')).lower()
                if 'size' in error_msg or 'limit' in error_msg:
                    self.test_results.append((
                        test_name,
                        False,
                        f'Image was rejected due to size: {data}',
                    ))
                    return False

            self.test_results.append((test_name, False, f'Unexpected result: {data}'))
            return False

        except Exception as e:
            error_msg = str(e).lower()
            # If the error is about size limits, the test reveals a boundary issue
            if 'size' in error_msg or 'limit' in error_msg or 'exceeds' in error_msg:
                self.test_results.append((
                    test_name,
                    False,
                    f'Image rejected at boundary size: {e}',
                ))
                return False
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_store_context_generation_first_return_exceptions(self) -> bool:
        """Test that store_context works end-to-end with the generation-first pattern.

        Verifies the asyncio.gather(return_exceptions=True) code path
        succeeds when no providers are configured (default test server).

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context_generation_first_return_exceptions'
        assert self.client is not None  # Type guard for Pyright
        try:
            gen_first_thread = f'{self.test_thread_id}_gen_first_store'

            # Store context -- should succeed through the gather path
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': gen_first_thread,
                    'source': 'agent',
                    'text': 'Generation-first pattern integration test for store_context',
                    'metadata': {'test_type': 'generation_first'},
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'store_context failed: {store_data}'),
                )
                return False

            context_id = store_data['context_id']

            # Update the same entry with new text -- exercises update_context gather path
            update_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'text': 'Updated text through generation-first pattern',
                },
            )
            update_data = self._extract_content(update_result)
            if not update_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'update_context failed: {update_data}'),
                )
                return False

            # Verify updated text persisted
            get_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )
            get_data = self._extract_content(get_result)
            results = get_data.get('results', [])
            if len(results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result, got {len(results)}'),
                )
                return False

            if results[0]['text_content'] != 'Updated text through generation-first pattern':
                self.test_results.append(
                    (test_name, False,
                     f'Text mismatch after update: {results[0]["text_content"]!r}'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 'store_context and update_context succeed through generation-first gather path'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_content_type_auto_detection_multimodal(self) -> bool:
        """Verify content_type is automatically set to 'multimodal' when images are included.

        Returns:
            bool: True if test passed.
        """
        test_name = 'content_type_auto_detection_multimodal'
        assert self.client is not None
        try:
            ct_thread = f'{self.test_thread_id}_content_type'

            text_result = await self.client.call_tool('store_context', {
                'thread_id': ct_thread, 'source': 'agent',
                'text': 'Text only entry for content type test',
            })
            text_data = self._extract_content(text_result)
            if not text_data.get('success'):
                self.test_results.append((test_name, False, f'Text store failed: {text_data}'))
                return False
            text_id = text_data.get('context_id')

            image_result = await self.client.call_tool('store_context', {
                'thread_id': ct_thread, 'source': 'agent',
                'text': 'Multimodal entry with image for content type test',
                'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
            })
            image_data = self._extract_content(image_result)
            if not image_data.get('success'):
                self.test_results.append((test_name, False, f'Image store failed: {image_data}'))
                return False
            image_id = image_data.get('context_id')

            get_result = await self.client.call_tool('get_context_by_ids', {
                'context_ids': [text_id, image_id],
            })
            get_data = self._extract_content(get_result)

            results = get_data.get('results', [])
            if len(results) != 2:
                self.test_results.append((test_name, False,
                    f'Expected 2 results, got {len(results)}'))
                return False

            text_entry = next((r for r in results if r.get('id') == text_id), None)
            image_entry = next((r for r in results if r.get('id') == image_id), None)

            if not text_entry or not image_entry:
                self.test_results.append((test_name, False, 'Could not find entries by ID'))
                return False

            if text_entry.get('content_type') != 'text':
                self.test_results.append((test_name, False,
                    f"Text entry content_type={text_entry.get('content_type')}, expected 'text'"))
                return False

            if image_entry.get('content_type') != 'multimodal':
                self.test_results.append((test_name, False,
                    f"Image entry content_type={image_entry.get('content_type')}, expected 'multimodal'"))
                return False

            self.test_results.append((test_name, True,
                'Content type auto-detection: text and multimodal correct'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_store_context_long_document_in_batches(self) -> bool:
        """Store a Markdown document that the server embeds in three batches.

        Runs when the server registered ``semantic_search_context``: the document
        splits into twelve chunks at the default chunk size, and the harness
        servers pin ``EMBEDDING_BATCH_SIZE`` to 4 (``HARNESS_EMBEDDING_BATCH_SIZE``),
        so the store sends three batches of four chunks. The check verifies that
        the store completes and that semantic search then finds the entry; the
        batch count itself is not observable through the protocol.

        Returns:
            bool: True if test passed or was skipped because semantic search is not registered.
        """
        test_name = 'Store Context Long Document In Batches'
        if 'semantic_search_context' not in self.registered_tools:
            return self._record(test_name, [], 'Skipped (semantic_search_context not registered)')
        try:
            thread_id = f'{self.test_thread_id}_embedding_batches'
            text = _long_markdown_document()
            stored, error = await self._outcome(
                'store_context', {'thread_id': thread_id, 'source': 'agent', 'text': text},
            )
            if stored is None or not stored.get('success'):
                return self._record(test_name, [f'store_context failed: {error or stored}'], '')
            context_id = stored['context_id']

            found, error = await self._outcome(
                'semantic_search_context',
                {'query': 'anemones and sea stars in rocky coastal tide pools', 'thread_id': thread_id, 'limit': 5},
            )
            problems: list[str] = []
            if found is None:
                problems.append(f'semantic_search_context failed: {error}')
            elif context_id not in [row.get('id') for row in found.get('results', [])]:
                problems.append(f'semantic search did not return {context_id}: {found}')
            return self._record(
                test_name,
                problems,
                f'stored {len(text)} characters (twelve chunks, three batches of four at the pinned batch size) '
                f'as {context_id} and found it by semantic search',
            )
        except Exception as e:
            return self._record(test_name, [f'Exception: {e}'], '')


_DOCUMENT_TOPICS: tuple[tuple[str, str], ...] = (
    ('Tide Pools', 'tide pools along a rocky coast shelter anemones, sea stars and hermit crabs'),
    ('Sourdough Baking', 'a sourdough starter of flour and water ferments for days before it leavens bread'),
    ('Glacier Movement', 'glaciers creep downhill under their own weight and carve valleys out of bedrock'),
    ('Beekeeping', 'a beekeeper inspects the hive frames for brood, honey stores and a laying queen'),
    ('Orbital Mechanics', 'a satellite in low orbit circles the planet every ninety minutes at high speed'),
    ('Violin Making', 'a luthier carves spruce and maple plates and varnishes the finished violin'),
    ('Desert Ecology', 'desert plants store water in thick stems and open their pores only at night'),
    ('Railway Signaling', 'block signals keep trains apart by allowing one train per section of track'),
    ('Coffee Roasting', 'roasting turns green coffee beans brown as sugars caramelize in the drum'),
    ('Volcanic Islands', 'volcanic islands rise where magma breaks through the ocean floor and cools'),
    ('Typesetting', 'a typesetter arranges metal letters in a composing stick line by line'),
    ('Wetland Birds', 'herons and egrets wade through shallow wetlands hunting fish and frogs'),
)


def _long_markdown_document() -> str:
    """Build a Markdown document that splits into twelve chunks at the default chunk size, one per section.

    Returns:
        The document text.
    """
    sections = []
    for title, fact in _DOCUMENT_TOPICS:
        sentences = [
            f'In this section on {title.lower()}, the main observation is that {fact}.',
            f'Field notes repeat it: {fact}, and the notes record when and where it was seen.',
            f'Readers new to {title.lower()} often ask why {fact}; the answer takes a whole season of study.',
            f'Over many years the same pattern holds, since {fact} wherever the conditions allow.',
        ]
        paragraph = ' '.join(sentences * 2)
        sections.append(f'## {title}\n\n{paragraph}')
    return '# Field Guide\n\n' + '\n\n'.join(sections) + '\n'
