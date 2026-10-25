"""Tests for app/settings/embedding.py.

Covers the EMBEDDING_QUERY_INSTRUCTION field, the Ollama and Voyage truncation
settings, the timeout bound and batch settings on EmbeddingSettings, and
ChunkingSettings.
"""

import pytest
from pydantic import ValidationError

from app.settings import AppSettings
from app.settings.embedding import ChunkingSettings
from tests.helpers import env_var
from tests.helpers import env_vars


class TestEmbeddingQueryInstruction:
    """Test the EMBEDDING_QUERY_INSTRUCTION field on EmbeddingSettings."""

    def test_query_instruction_default_none(self) -> None:
        """The instruction defaults to None so query embedding text stays bare."""
        from app.settings.embedding import EmbeddingSettings

        with env_var('EMBEDDING_QUERY_INSTRUCTION', None):
            settings = EmbeddingSettings()
        assert settings.query_instruction is None

    def test_query_instruction_field_alias(self) -> None:
        """The field maps to the EMBEDDING_QUERY_INSTRUCTION environment variable."""
        from app.settings.embedding import EmbeddingSettings

        field_info = EmbeddingSettings.model_fields['query_instruction']
        assert field_info.alias == 'EMBEDDING_QUERY_INSTRUCTION'

    def test_query_instruction_env_value_preserved_verbatim(self) -> None:
        """A multi-line env value survives verbatim, including the embedded newline."""
        from app.settings.embedding import EmbeddingSettings

        value = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
        with env_var('EMBEDDING_QUERY_INSTRUCTION', value):
            settings = EmbeddingSettings()
        assert settings.query_instruction == value

    def test_query_instruction_empty_env_is_falsy(self) -> None:
        """An empty env value stays falsy so the query path treats it as unset."""
        from app.settings.embedding import EmbeddingSettings

        with env_var('EMBEDDING_QUERY_INSTRUCTION', ''):
            settings = EmbeddingSettings()
        assert not settings.query_instruction

    def test_query_instruction_via_app_settings(self) -> None:
        """The env value propagates through AppSettings.embedding."""
        with env_var('EMBEDDING_QUERY_INSTRUCTION', 'Prefix: '):
            settings = AppSettings()
        assert settings.embedding.query_instruction == 'Prefix: '


class TestOllamaTruncateSettings:
    """Test EMBEDDING_OLLAMA_TRUNCATE and EMBEDDING_OLLAMA_NUM_CTX settings."""

    def test_ollama_truncate_default_is_false(self) -> None:
        """Verify EMBEDDING_OLLAMA_TRUNCATE defaults to false (prevent silent truncation)."""
        with env_vars(EMBEDDING_OLLAMA_TRUNCATE=None, EMBEDDING_OLLAMA_NUM_CTX=None):
            settings = AppSettings()
            assert settings.embedding.ollama_truncate is False

    def test_ollama_truncate_can_be_set_true(self) -> None:
        """Verify EMBEDDING_OLLAMA_TRUNCATE can be explicitly set to true."""
        with env_vars(EMBEDDING_OLLAMA_TRUNCATE='true'):
            settings = AppSettings()
            assert settings.embedding.ollama_truncate is True

    def test_ollama_truncate_can_be_set_false(self) -> None:
        """Verify EMBEDDING_OLLAMA_TRUNCATE can be explicitly set to false."""
        with env_vars(EMBEDDING_OLLAMA_TRUNCATE='false'):
            settings = AppSettings()
            assert settings.embedding.ollama_truncate is False

    def test_ollama_num_ctx_default_is_4096(self) -> None:
        """Verify EMBEDDING_OLLAMA_NUM_CTX defaults to 4096."""
        with env_vars(EMBEDDING_OLLAMA_NUM_CTX=None):
            settings = AppSettings()
            assert settings.embedding.ollama_num_ctx == 4096

    def test_ollama_num_ctx_can_be_customized(self) -> None:
        """Verify EMBEDDING_OLLAMA_NUM_CTX can be set to custom value."""
        with env_vars(EMBEDDING_OLLAMA_NUM_CTX='8192'):
            settings = AppSettings()
            assert settings.embedding.ollama_num_ctx == 8192

    def test_ollama_num_ctx_minimum_validation(self) -> None:
        """Verify EMBEDDING_OLLAMA_NUM_CTX validates minimum value (512)."""
        with env_vars(EMBEDDING_OLLAMA_NUM_CTX='100'), pytest.raises(ValidationError):
            AppSettings()

    def test_ollama_num_ctx_maximum_validation(self) -> None:
        """Verify EMBEDDING_OLLAMA_NUM_CTX validates maximum value (2097152)."""
        with env_vars(EMBEDDING_OLLAMA_NUM_CTX='3000000'), pytest.raises(ValidationError):
            AppSettings()


class TestVoyageTruncationSettings:
    """Test VOYAGE_TRUNCATION settings."""

    def test_voyage_truncation_default_is_false(self) -> None:
        """Verify VOYAGE_TRUNCATION defaults to false (prevent silent truncation)."""
        with env_vars(VOYAGE_TRUNCATION=None):
            settings = AppSettings()
            assert settings.embedding.voyage_truncation is False

    def test_voyage_truncation_can_be_set_true(self) -> None:
        """Verify VOYAGE_TRUNCATION can be explicitly set to true."""
        with env_vars(VOYAGE_TRUNCATION='true'):
            settings = AppSettings()
            assert settings.embedding.voyage_truncation is True

    def test_voyage_truncation_can_be_set_false(self) -> None:
        """Verify VOYAGE_TRUNCATION can be explicitly set to false."""
        with env_vars(VOYAGE_TRUNCATION='false'):
            settings = AppSettings()
            assert settings.embedding.voyage_truncation is False


class TestEmbeddingBatchSettings:
    """Test EMBEDDING_TIMEOUT_S bounds and the EMBEDDING_BATCH_* settings."""

    def test_defaults(self) -> None:
        """Batch sizing defaults to auto with a five-second target; the timeout stays 240 s."""
        from app.settings.embedding import EmbeddingSettings

        with env_vars(EMBEDDING_BATCH_SIZE=None, EMBEDDING_BATCH_TARGET_S=None, EMBEDDING_TIMEOUT_S=None):
            settings = EmbeddingSettings()
        assert settings.batch_size == 'auto'
        assert settings.batch_target_s == 5.0
        assert settings.timeout_s == 240.0

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [('8', 8), ('1', 1), ('512', 512), (' 16 ', 16), ('auto', 'auto'), ('AUTO', 'auto'), (' Auto ', 'auto')],
    )
    def test_batch_size_accepts_auto_and_integers(self, raw: str, expected: int | str) -> None:
        """A digit string becomes an integer and any casing of auto becomes 'auto'."""
        from app.settings.embedding import EmbeddingSettings

        with env_var('EMBEDDING_BATCH_SIZE', raw):
            settings = EmbeddingSettings()
        assert settings.batch_size == expected

    @pytest.mark.parametrize('raw', ['0', '513', '-1', 'x', '2.5', ''])
    def test_batch_size_rejects_other_values(self, raw: str) -> None:
        """Values outside 1..512 and words other than auto are configuration errors."""
        from app.settings.embedding import EmbeddingSettings

        with env_var('EMBEDDING_BATCH_SIZE', raw), pytest.raises(ValidationError):
            EmbeddingSettings()

    @pytest.mark.parametrize(('raw', 'expected'), [('0.5', 0.5), ('3600', 3600.0)])
    def test_batch_target_accepts_its_range(self, raw: str, expected: float) -> None:
        """The batch target accepts any positive value up to an hour."""
        from app.settings.embedding import EmbeddingSettings

        with env_var('EMBEDDING_BATCH_TARGET_S', raw):
            settings = EmbeddingSettings()
        assert settings.batch_target_s == expected

    @pytest.mark.parametrize('raw', ['0', '-1', '3600.5'])
    def test_batch_target_rejects_values_outside_its_range(self, raw: str) -> None:
        """A zero, negative or over-an-hour target is a configuration error."""
        from app.settings.embedding import EmbeddingSettings

        with env_var('EMBEDDING_BATCH_TARGET_S', raw), pytest.raises(ValidationError):
            EmbeddingSettings()

    def test_timeout_accepts_up_to_an_hour(self) -> None:
        """EMBEDDING_TIMEOUT_S accepts 3600 and refuses anything above it."""
        from app.settings.embedding import EmbeddingSettings

        with env_var('EMBEDDING_TIMEOUT_S', '3600'):
            assert EmbeddingSettings().timeout_s == 3600.0
        with env_var('EMBEDDING_TIMEOUT_S', '3600.5'), pytest.raises(ValidationError):
            EmbeddingSettings()


class TestChunkingSettings:
    """Tests for ChunkingSettings validation."""

    def test_default_values(self) -> None:
        """Default values should be valid."""
        settings = ChunkingSettings()
        assert settings.enabled is True
        assert settings.size == 1500
        assert settings.overlap == 150
        assert settings.aggregation == 'max'

    def test_overlap_must_be_less_than_size(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Overlap must be strictly less than chunk size."""
        monkeypatch.setenv('CHUNK_SIZE', '100')
        monkeypatch.setenv('CHUNK_OVERLAP', '100')
        with pytest.raises(ValidationError) as exc_info:
            ChunkingSettings()
        assert 'CHUNK_OVERLAP' in str(exc_info.value)
        assert 'must be less than' in str(exc_info.value)

    def test_overlap_greater_than_size_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Overlap greater than size should fail."""
        monkeypatch.setenv('CHUNK_SIZE', '100')
        monkeypatch.setenv('CHUNK_OVERLAP', '150')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_valid_overlap_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid overlap should pass validation."""
        monkeypatch.setenv('CHUNK_SIZE', '500')
        monkeypatch.setenv('CHUNK_OVERLAP', '100')
        settings = ChunkingSettings()
        assert settings.size == 500
        assert settings.overlap == 100

    def test_size_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimum valid chunk size should pass."""
        monkeypatch.setenv('CHUNK_SIZE', '100')
        monkeypatch.setenv('CHUNK_OVERLAP', '50')
        settings = ChunkingSettings()
        assert settings.size == 100

    def test_size_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Maximum valid chunk size should pass."""
        monkeypatch.setenv('CHUNK_SIZE', '10000')
        monkeypatch.setenv('CHUNK_OVERLAP', '100')
        settings = ChunkingSettings()
        assert settings.size == 10000

    def test_size_below_minimum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chunk size below minimum should fail."""
        monkeypatch.setenv('CHUNK_SIZE', '99')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_size_above_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chunk size above maximum should fail."""
        monkeypatch.setenv('CHUNK_SIZE', '10001')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_overlap_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimum valid overlap (0) should pass."""
        monkeypatch.setenv('CHUNK_OVERLAP', '0')
        settings = ChunkingSettings()
        assert settings.overlap == 0

    def test_overlap_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Maximum valid overlap should pass."""
        monkeypatch.setenv('CHUNK_SIZE', '1000')
        monkeypatch.setenv('CHUNK_OVERLAP', '500')
        settings = ChunkingSettings()
        assert settings.overlap == 500

    def test_overlap_negative_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative overlap should fail."""
        monkeypatch.setenv('CHUNK_OVERLAP', '-1')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_overlap_above_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Overlap above maximum should fail."""
        monkeypatch.setenv('CHUNK_OVERLAP', '501')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_aggregation_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Aggregation 'max' should be valid."""
        monkeypatch.setenv('CHUNK_AGGREGATION', 'max')
        settings = ChunkingSettings()
        assert settings.aggregation == 'max'

    def test_aggregation_invalid_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid aggregation (including 'avg' and 'sum') should fail."""
        monkeypatch.setenv('CHUNK_AGGREGATION', 'invalid')
        with pytest.raises(ValidationError):
            ChunkingSettings()

    def test_environment_variable_aliases(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Settings should read from environment variables."""
        monkeypatch.setenv('ENABLE_CHUNKING', 'false')
        monkeypatch.setenv('CHUNK_SIZE', '2000')
        monkeypatch.setenv('CHUNK_OVERLAP', '200')
        monkeypatch.setenv('CHUNK_AGGREGATION', 'max')

        settings = ChunkingSettings()
        assert settings.enabled is False
        assert settings.size == 2000
        assert settings.overlap == 200
        assert settings.aggregation == 'max'
