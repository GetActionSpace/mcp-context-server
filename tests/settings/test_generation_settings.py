"""Tests for app/settings/generation.py.

Covers GENERATION_JOB_MAX_ATTEMPTS, the number of attempts a deferred generation
job gets: its default, its alias, its 1-10 range, and its place on AppSettings.
"""

import pytest
from pydantic import ValidationError

from app.settings import get_settings
from app.settings.generation import GenerationJobSettings


class TestGenerationJobMaxAttempts:
    """GENERATION_JOB_MAX_ATTEMPTS defaults to 3 and accepts 1 through 10."""

    def test_default_is_three(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('GENERATION_JOB_MAX_ATTEMPTS', raising=False)
        assert GenerationJobSettings().max_attempts == 3

    def test_field_reads_the_generation_job_max_attempts_variable(self) -> None:
        assert GenerationJobSettings.model_fields['max_attempts'].alias == 'GENERATION_JOB_MAX_ATTEMPTS'

    def test_environment_value_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GENERATION_JOB_MAX_ATTEMPTS', '7')
        assert GenerationJobSettings().max_attempts == 7

    @pytest.mark.parametrize('value', ['1', '10'])
    def test_range_ends_are_accepted(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv('GENERATION_JOB_MAX_ATTEMPTS', value)
        assert GenerationJobSettings().max_attempts == int(value)

    @pytest.mark.parametrize('value', ['0', '11', '-1'])
    def test_values_outside_the_range_are_refused(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv('GENERATION_JOB_MAX_ATTEMPTS', value)
        with pytest.raises(ValidationError, match='GENERATION_JOB_MAX_ATTEMPTS'):
            GenerationJobSettings()

    def test_app_settings_composes_it_as_generation_jobs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GENERATION_JOB_MAX_ATTEMPTS', '2')
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings.generation_jobs, GenerationJobSettings)
        assert settings.generation_jobs.max_attempts == 2
