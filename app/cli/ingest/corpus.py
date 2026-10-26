"""The corpus a sync client reads: the files selected under a directory and the documents they become.

A corpus is a directory tree. Files are selected by glob patterns matched against
their root-relative POSIX paths, case-sensitively on every host, so a Windows host
and a Linux container select the same files. Each selected file becomes a document
whose text is the file decoded as UTF-8 with LF line endings and without
surrounding whitespace, the form the server stores and hashes, and whose source id
is its root-relative POSIX path in NFC.

A file the client cannot use in this run never turns into a deletion of what the
server holds. An unreadable, undecodable, empty or oversized file is kept: the
manifest names it with a keep item and its stored entry stays as it is. A directory
the walk cannot list is recorded, because the files under it are missing from the
manifest and the run must not delete their entries. A file whose source id exceeds
the protocol bound is excluded from the manifest, because no stored entry can carry
such an id.
"""

import functools
import operator
import os
import re
import unicodedata
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Literal

from app.cli.ingest.manifest import meta_digest
from app.cli.ingest.manifest import request_bytes
from app.cli.ingest.manifest import text_digest
from app.digests import normalize_stored_text
from app.ingest import INGEST_KEY
from app.ingest import MAX_SOURCE_ID_LENGTH
from app.ingest import SyncChange
from app.models import normalize_tag_list

type Visibility = Literal['private', 'public']

# Knowledge-base entries are stored as user content: the corpus is what a person wrote.
ENTRY_SOURCE = 'user'
GIT_DIRECTORY = '.git'
_SYNCED_AT_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

_LEVEL_ONE_HEADING = re.compile(r'[ ]{0,3}#[ \t]+(.*)')
# A closing sequence of '#' characters ends an ATX heading when whitespace precedes it
# or when it is the whole heading text.
_CLOSING_SEQUENCE = re.compile(r'(?:^|[ \t]+)#+[ \t]*$')


def format_synced_at(moment: datetime) -> str:
    """Return the ``synced_at`` value of the ingest block for a run started at a moment.

    Args:
        moment: The run's start time, timezone-aware.

    Returns:
        The moment in UTC with whole seconds, such as ``2026-10-18T12:00:00Z``.
    """
    return moment.astimezone(UTC).strftime(_SYNCED_AT_FORMAT)


# Every run time format_synced_at writes and every context id the server issues has a
# fixed width, so these stand-ins size an entry exactly before the run sends it.
_SIZING_SYNCED_AT = format_synced_at(datetime(2000, 1, 1, tzinfo=UTC))
_SIZING_CONTEXT_ID = '0' * 32


@dataclass(frozen=True)
class Document:
    """One selected file as the client sends it.

    Attributes:
        source_id: The root-relative POSIX path in NFC, the document's identity in the corpus.
        text: The stored form of the file's text.
        metadata: The metadata the client controls, without the ingest block.
        tags: The normalized tags sent with every create and update.
        visibility: The visibility sent with every create and update.
        defer_generation: Whether every create and update asks the server to store the
            entry at once and generate its embeddings and summaries in the background.
        text_digest: The digest the server records for the stored text.
        meta_digest: The digest of the metadata, the tags and the visibility.
    """

    source_id: str
    text: str
    metadata: dict[str, object]
    tags: list[str]
    visibility: Visibility
    defer_generation: bool
    text_digest: str
    meta_digest: str

    def entry_metadata(self, *, thread_id: str, synced_at: str) -> dict[str, object]:
        """Return the metadata a create or update sends: the document metadata plus the ingest block.

        Args:
            thread_id: The corpus thread, which is the block's scope.
            synced_at: The run time from :func:`format_synced_at`.

        Returns:
            A new metadata object.
        """
        block: dict[str, object] = {
            'scope': thread_id,
            'source_id': self.source_id,
            'meta_digest': self.meta_digest,
            'synced_at': synced_at,
        }
        return {**self.metadata, INGEST_KEY: block}

    def create_entry(self, *, thread_id: str, synced_at: str) -> dict[str, object]:
        """Return the ``store_context_batch`` entry that stores the document as a new entry.

        Deduplication is off, so the store always inserts one entry owned by the caller.
        A deferring document's entry also asks the server to defer generation.

        Args:
            thread_id: The corpus thread.
            synced_at: The run time from :func:`format_synced_at`.

        Returns:
            A new batch entry.
        """
        entry: dict[str, object] = {
            'thread_id': thread_id,
            'source': ENTRY_SOURCE,
            'text': self.text,
            'metadata': self.entry_metadata(thread_id=thread_id, synced_at=synced_at),
            'tags': list(self.tags),
            'visibility': self.visibility,
            'deduplicate': False,
        }
        if self.defer_generation:
            entry['defer_generation'] = True
        return entry

    def update_entry(self, *, context_id: str, thread_id: str, synced_at: str, change: SyncChange) -> dict[str, object]:
        """Return the ``update_context_batch`` entry that brings a stored entry up to the document.

        The metadata is always replaced in full, never patched, so the stored block
        and the digest it carries always describe the whole object; the text is sent
        only when it changed. A deferring document's entry also asks the server to
        defer generation, which the server applies only to an update that sends text.

        Args:
            context_id: The stored entry to update.
            thread_id: The corpus thread.
            synced_at: The run time from :func:`format_synced_at`.
            change: What the plan found different.

        Returns:
            A new batch entry.
        """
        entry: dict[str, object] = {'context_id': context_id}
        if change != 'metadata':
            entry['text'] = self.text
        entry['metadata'] = self.entry_metadata(thread_id=thread_id, synced_at=synced_at)
        entry['tags'] = list(self.tags)
        entry['visibility'] = self.visibility
        if self.defer_generation:
            entry['defer_generation'] = True
        return entry


@dataclass(frozen=True)
class FileFailure:
    """A selected file that did not become a document in this run.

    Attributes:
        source_id: The file's source id.
        path: The file.
        reason: Why the file was not used.
        keep: True when a keep item represents the file in the manifest, so its
            stored entry survives; False when the file is excluded from the manifest.
    """

    source_id: str
    path: Path
    reason: str
    keep: bool


@dataclass(frozen=True)
class DirectoryFailure:
    """A directory the walk could not list, so the files under it are missing from the run.

    Attributes:
        relative_path: The directory's root-relative POSIX path, ``'.'`` for the root.
        reason: The operating system's error message.
    """

    relative_path: str
    reason: str


@dataclass(frozen=True)
class Corpus:
    """Everything a run read from its source directory.

    Attributes:
        documents: The documents, sorted by source id.
        file_failures: The selected files that did not become documents, sorted by source id.
        directory_failures: The directories the walk could not list.
    """

    documents: list[Document]
    file_failures: list[FileFailure]
    directory_failures: list[DirectoryFailure]


class _UnusableFileError(Exception):
    """A selected file whose content cannot become a document in this run."""


def matches(pattern: str, relative_posix_path: str) -> bool:
    """Return whether a glob pattern matches a root-relative POSIX path.

    Matching is case-sensitive on every host. A ``**`` segment matches any number
    of path segments, ``**/`` none included; ``*``, ``?`` and character classes
    never match a slash; classes follow ``fnmatch`` syntax, with ``!`` negating
    and a leading ``]`` taken literally. A class whose range runs backwards cannot
    be compiled and raises ``ValueError``. Each pattern is translated once.

    Args:
        pattern: The glob pattern.
        relative_posix_path: The path to test, relative to the corpus root.

    Returns:
        True when the pattern matches the whole path.
    """
    return _compile_glob(pattern).fullmatch(relative_posix_path) is not None


def walk(
    root: Path,
    *,
    includes: Sequence[str],
    excludes: Sequence[str],
) -> tuple[list[tuple[str, Path]], list[DirectoryFailure]]:
    """Select the files under a root that an include matches and no exclude matches.

    ``.git`` directories are never entered, symbolic links are never followed, and a
    directory that cannot be listed is recorded instead of skipped silently.

    Args:
        root: The corpus root.
        includes: Patterns of which at least one must match a file.
        excludes: Patterns of which none may match a file.

    Returns:
        The selected files as ``(source_id, path)`` pairs sorted by source id, and
        the directories that could not be listed.
    """
    failures: list[DirectoryFailure] = []

    def record(error: OSError) -> None:
        failures.append(DirectoryFailure(relative_path=_failed_directory(root, error), reason=error.strerror or str(error)))

    selected: list[tuple[str, Path]] = []
    for directory, dirnames, filenames in os.walk(root, onerror=record, followlinks=False):
        dirnames[:] = [name for name in dirnames if name != GIT_DIRECTORY]
        for name in filenames:
            path = Path(directory, name)
            if path.is_symlink():
                continue
            source_id = _relative_posix(root, path)
            if any(matches(pattern, source_id) for pattern in includes) and not any(
                matches(pattern, source_id) for pattern in excludes
            ):
                selected.append((source_id, path))
    selected.sort(key=operator.itemgetter(0))
    return selected, failures


def build_corpus(
    root: Path,
    *,
    thread_id: str,
    includes: Sequence[str],
    excludes: Sequence[str],
    static_metadata: Mapping[str, object],
    tags: Sequence[str],
    visibility: Visibility,
    defer_generation: bool,
    max_request_bytes: int,
) -> Corpus:
    """Walk a source directory and turn every selected file into a document or a failure.

    Args:
        root: The corpus root.
        thread_id: The corpus thread, written into each document's ingest block.
        includes: Patterns of which at least one must match a file.
        excludes: Patterns of which none may match a file.
        static_metadata: Metadata applied to every document; its ``title`` replaces
            the one derived from the file.
        tags: Tags applied to every document.
        visibility: The visibility sent with every create and update.
        defer_generation: Whether every create and update asks the server to defer generation.
        max_request_bytes: The request budget; a document whose batch entry alone
            exceeds it is kept instead of sent.

    Returns:
        The documents, the files that did not become documents and the directories
        that could not be listed.
    """
    files, directory_failures = walk(root, includes=includes, excludes=excludes)
    normalized_tags = normalize_tag_list(list(tags))
    documents: list[Document] = []
    file_failures: list[FileFailure] = []
    for source_id, path in files:
        if len(source_id) > MAX_SOURCE_ID_LENGTH:
            reason = f'the source id is {len(source_id)} characters long, above the limit of {MAX_SOURCE_ID_LENGTH}'
            file_failures.append(FileFailure(source_id=source_id, path=path, reason=reason, keep=False))
            continue
        try:
            text = _read_text(path)
        except _UnusableFileError as error:
            file_failures.append(FileFailure(source_id=source_id, path=path, reason=str(error), keep=True))
            continue
        metadata: dict[str, object] = {'title': _title(text, source_id), **static_metadata}
        document = Document(
            source_id=source_id,
            text=text,
            metadata=metadata,
            tags=list(normalized_tags),
            visibility=visibility,
            defer_generation=defer_generation,
            text_digest=text_digest(text),
            meta_digest=meta_digest(metadata, normalized_tags, visibility),
        )
        size = _largest_request(document, thread_id=thread_id)
        if size > max_request_bytes:
            reason = (
                f'a batch request carrying it needs {size:,} bytes, above the --max-request-bytes budget of '
                f"{max_request_bytes:,} bytes (docs/knowledge-base.md, 'Limits')"
            )
            file_failures.append(FileFailure(source_id=source_id, path=path, reason=reason, keep=True))
            continue
        documents.append(document)
    return Corpus(documents=documents, file_failures=file_failures, directory_failures=directory_failures)


@functools.lru_cache(maxsize=256)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Translate a glob pattern into a compiled, case-sensitive regular expression."""
    parts: list[str] = []
    index, length = 0, len(pattern)
    while index < length:
        char = pattern[index]
        if char == '*':
            end = index
            while end < length and pattern[end] == '*':
                end += 1
            starts_segment = index == 0 or pattern[index - 1] == '/'
            ends_segment = end == length or pattern[end] == '/'
            if end - index >= 2 and starts_segment and ends_segment:
                if end < length:
                    parts.append('(?:[^/]*/)*')
                    end += 1
                else:
                    parts.append('.*')
            else:
                parts.append('[^/]*')
            index = end
        elif char == '?':
            parts.append('[^/]')
            index += 1
        elif char == '[' and (translated := _character_class(pattern, index + 1)) is not None:
            class_regex, index = translated
            parts.append(class_regex)
        else:
            parts.append(re.escape(char))
            index += 1
    try:
        return re.compile(''.join(parts), re.DOTALL)
    except re.error as error:
        raise ValueError(f'invalid glob pattern {pattern!r}: {error}') from error


def _character_class(pattern: str, start: int) -> tuple[str, int] | None:
    """Translate the character class whose bracket opens just before ``start``.

    Returns:
        The class as a regular expression that never matches a slash and the index
        past the closing bracket, or None when the bracket is never closed and so
        stands for itself.
    """
    end = start
    if end < len(pattern) and pattern[end] == '!':
        end += 1
    if end < len(pattern) and pattern[end] == ']':
        end += 1
    while end < len(pattern) and pattern[end] != ']':
        end += 1
    if end >= len(pattern):
        return None
    body = pattern[start:end]
    negated = body.startswith('!')
    if negated:
        body = body[1:]
    # Backslashes and the characters a regular-expression class treats specially are
    # literal in a glob class; ranges keep their hyphens.
    body = re.sub(r'([\\&~|^\[\]])', r'\\\1', body)
    return f'(?!/)[{"^" if negated else ""}{body}]', end + 1


def _relative_posix(root: Path, path: Path) -> str:
    """Return a path relative to the root as a POSIX path in NFC."""
    return unicodedata.normalize('NFC', path.relative_to(root).as_posix())


def _failed_directory(root: Path, error: OSError) -> str:
    """Return the root-relative POSIX path of the directory a listing error names."""
    if error.filename is None:
        return '.'
    return _relative_posix(root, Path(os.fsdecode(error.filename)))


def _read_text(path: Path) -> str:
    """Return a file's text in the form the server stores.

    Returns:
        The text decoded as UTF-8, with LF line endings and without surrounding whitespace.

    Raises:
        _UnusableFileError: When the file cannot be read, is not UTF-8, or holds no
            text once surrounding whitespace is removed.
    """
    try:
        data = path.read_bytes()
    except OSError as error:
        raise _UnusableFileError(f'cannot read the file: {error.strerror or error}') from error
    try:
        decoded = data.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise _UnusableFileError(f'the file is not valid UTF-8: {error}') from error
    text = normalize_stored_text(decoded.replace('\r\n', '\n').replace('\r', '\n'))
    if not text:
        raise _UnusableFileError('the text is empty once surrounding whitespace is removed')
    return text


def _title(text: str, source_id: str) -> str:
    """Return the text of the first ATX level-one heading that has text, else the file stem."""
    for line in text.split('\n'):
        heading = _LEVEL_ONE_HEADING.fullmatch(line)
        if heading is not None and (title := _CLOSING_SEQUENCE.sub('', heading.group(1).strip()).strip()):
            return title
    return PurePosixPath(source_id).stem


def _largest_request(document: Document, *, thread_id: str) -> int:
    """Return the size of the larger of the document's create and full update requests."""
    create = document.create_entry(thread_id=thread_id, synced_at=_SIZING_SYNCED_AT)
    update = document.update_entry(
        context_id=_SIZING_CONTEXT_ID,
        thread_id=thread_id,
        synced_at=_SIZING_SYNCED_AT,
        change='both',
    )
    return max(request_bytes([create]), request_bytes([update]))
