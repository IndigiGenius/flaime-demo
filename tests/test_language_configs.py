"""Tests for the demo YAML routing configs (DEMO-04, moved in 26Q3-REPO-11).

Validates that the three shipped config files load correctly via LanguageRouter
and produce the expected routing decisions without touching any model weights.

Configs under test:
  configs/demo_languages.yaml             — 64-lang XEUS set; hu has expert
  configs/demo_languages_merged_only.yaml — all 64 XEUS langs via merged
  configs/demo_languages_wav2vec2.yaml    — 3-lang wav2vec2 baseline
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flaime_serving import LanguageNotSupportedError, LanguageRouter

MERGED_CHECKPOINT = (
    "./checkpoints/26Q2-XEUS-BTM-03_xeus-tiny-ssl_64lang_merged-average_s2055460031.pt"
)
HUNGARIAN_EXPERT = (
    "./checkpoints/26Q2-XEUS-BTM-03_xeus-tiny-ssl_hung1274_expert_ep9_s2055460031.pt"
)


# ── Config paths ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    """Routing YAMLs, resolved against the flaime-demo repo root.

    Via the `repo_root` fixture in conftest.py (pytest's own rootdir) rather
    than `Path(__file__).parents[N]`, which is banned in this codebase and
    would silently break if this file ever moved a directory.
    """
    return repo_root / "configs"


@pytest.fixture(scope="session")
def demo_yaml(configs_dir: Path) -> Path:
    return configs_dir / "demo_languages.yaml"


@pytest.fixture(scope="session")
def merged_only_yaml(configs_dir: Path) -> Path:
    return configs_dir / "demo_languages_merged_only.yaml"


@pytest.fixture(scope="session")
def wav2vec2_yaml(configs_dir: Path) -> Path:
    return configs_dir / "demo_languages_wav2vec2.yaml"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def demo_router(demo_yaml: Path) -> LanguageRouter:
    return LanguageRouter(demo_yaml)


@pytest.fixture(scope="module")
def merged_router(merged_only_yaml: Path) -> LanguageRouter:
    return LanguageRouter(merged_only_yaml)


@pytest.fixture(scope="module")
def wav2vec2_router(wav2vec2_yaml: Path) -> LanguageRouter:
    return LanguageRouter(wav2vec2_yaml)


# ── Config file existence ──────────────────────────────────────────────────────


class TestConfigFilesExist:
    def test_demo_languages_yaml_exists(self, demo_yaml: Path) -> None:
        assert demo_yaml.exists()

    def test_merged_only_yaml_exists(self, merged_only_yaml: Path) -> None:
        assert merged_only_yaml.exists()

    def test_wav2vec2_yaml_exists(self, wav2vec2_yaml: Path) -> None:
        assert wav2vec2_yaml.exists()


# ── demo_languages.yaml ───────────────────────────────────────────────────────


class TestDemoLanguagesYaml:
    def test_loads_without_error(self, demo_router: LanguageRouter) -> None:
        assert demo_router is not None

    def test_merged_checkpoint_path(self, demo_router: LanguageRouter) -> None:
        route = demo_router.resolve("es")
        assert route.checkpoint_path == MERGED_CHECKPOINT

    def test_merged_checkpoint_model_type_xeus(
        self, demo_router: LanguageRouter
    ) -> None:
        route = demo_router.resolve("es")
        assert route.model_type == "xeus"

    def test_hungarian_uses_expert_checkpoint(
        self, demo_router: LanguageRouter
    ) -> None:
        route = demo_router.resolve("hu")
        assert route.checkpoint_type == "expert"
        assert route.checkpoint_path == HUNGARIAN_EXPERT

    def test_hungarian_model_type_xeus(self, demo_router: LanguageRouter) -> None:
        assert demo_router.resolve("hu").model_type == "xeus"

    def test_english_falls_back_to_merged(self, demo_router: LanguageRouter) -> None:
        # DEMO-05 finding: the previous en expert override pointed at a
        # nonexistent placeholder ("./checkpoints/test_serving") and crashed
        # every English request. en routes via merged until a real expert ships.
        route = demo_router.resolve("en")
        assert route.model_type == "xeus"
        assert route.checkpoint_type == "merged"

    def test_non_expert_language_uses_merged(self, demo_router: LanguageRouter) -> None:
        for code in ("es", "fr", "ja", "sw", "zu"):
            route = demo_router.resolve(code)
            assert route.checkpoint_type == "merged", (
                f"{code} should route to merged, got {route.checkpoint_type}"
            )

    def test_only_hu_has_expert_checkpoint(self, demo_router: LanguageRouter) -> None:
        expert_langs = [
            code
            for code in demo_router.supported_languages()
            if demo_router.resolve(code).checkpoint_type == "expert"
        ]
        assert set(expert_langs) == {"hu"}

    def test_all_xeus_languages_use_merged_or_expert(
        self, demo_router: LanguageRouter
    ) -> None:
        for code, _name in demo_router.supported_languages().items():
            route = demo_router.resolve(code)
            assert route.model_type == "xeus", (
                f"{code} should use xeus, got {route.model_type}"
            )

    def test_unsupported_language_raises(self, demo_router: LanguageRouter) -> None:
        with pytest.raises(LanguageNotSupportedError):
            demo_router.resolve("xyz")

    def test_african_languages_present(self, demo_router: LanguageRouter) -> None:
        for code in ("sw", "am", "yo", "zu", "xh", "af", "rw"):
            assert code in demo_router.supported_languages(), (
                f"African language {code} missing from demo_languages.yaml"
            )

    def test_asian_languages_present(self, demo_router: LanguageRouter) -> None:
        for code in ("ja", "ko", "zh", "vi", "id", "th", "hi", "bn"):
            assert code in demo_router.supported_languages()

    def test_european_languages_present(self, demo_router: LanguageRouter) -> None:
        for code in ("es", "fr", "de", "it", "pl", "ru", "uk", "cs"):
            assert code in demo_router.supported_languages()


# ── demo_languages_merged_only.yaml ──────────────────────────────────────────


class TestMergedOnlyYaml:
    def test_loads_without_error(self, merged_router: LanguageRouter) -> None:
        assert merged_router is not None

    def test_merged_checkpoint_path(self, merged_router: LanguageRouter) -> None:
        route = merged_router.resolve("hu")
        assert route.checkpoint_path == MERGED_CHECKPOINT

    def test_no_expert_checkpoints(self, merged_router: LanguageRouter) -> None:
        for code in merged_router.supported_languages():
            route = merged_router.resolve(code)
            assert route.checkpoint_type == "merged", (
                f"{code} should use merged in merged-only config, "
                f"got {route.checkpoint_type}"
            )

    def test_hungarian_uses_merged(self, merged_router: LanguageRouter) -> None:
        route = merged_router.resolve("hu")
        assert route.checkpoint_type == "merged"
        assert route.checkpoint_path == MERGED_CHECKPOINT

    def test_all_languages_model_type_xeus(self, merged_router: LanguageRouter) -> None:
        for code in merged_router.supported_languages():
            route = merged_router.resolve(code)
            assert route.model_type == "xeus", (
                f"{code} should use xeus in merged-only config"
            )

    def test_english_not_in_merged_only(self, merged_router: LanguageRouter) -> None:
        with pytest.raises(LanguageNotSupportedError):
            merged_router.resolve("en")

    def test_language_count(self, merged_router: LanguageRouter) -> None:
        count = len(merged_router.supported_languages())
        assert count > 60, f"Expected 60+ languages in merged-only config, got {count}"

    def test_same_checkpoint_for_all(self, merged_router: LanguageRouter) -> None:
        paths = {
            merged_router.resolve(code).checkpoint_path
            for code in merged_router.supported_languages()
        }
        assert paths == {MERGED_CHECKPOINT}, (
            "All languages must resolve to the single merged checkpoint"
        )


# ── demo_languages_wav2vec2.yaml ──────────────────────────────────────────────


class TestWav2Vec2Yaml:
    def test_loads_without_error(self, wav2vec2_router: LanguageRouter) -> None:
        assert wav2vec2_router is not None

    def test_has_english(self, wav2vec2_router: LanguageRouter) -> None:
        assert "en" in wav2vec2_router.supported_languages()

    def test_has_spanish(self, wav2vec2_router: LanguageRouter) -> None:
        assert "es" in wav2vec2_router.supported_languages()

    def test_has_french(self, wav2vec2_router: LanguageRouter) -> None:
        assert "fr" in wav2vec2_router.supported_languages()

    def test_exactly_three_languages(self, wav2vec2_router: LanguageRouter) -> None:
        assert len(wav2vec2_router.supported_languages()) == 3

    def test_all_use_wav2vec2(self, wav2vec2_router: LanguageRouter) -> None:
        for code in wav2vec2_router.supported_languages():
            route = wav2vec2_router.resolve(code)
            assert route.model_type == "wav2vec2", f"{code} should use wav2vec2"

    def test_all_use_expert_checkpoint(self, wav2vec2_router: LanguageRouter) -> None:
        for code in wav2vec2_router.supported_languages():
            route = wav2vec2_router.resolve(code)
            assert route.checkpoint_type == "expert", (
                f"{code} should have expert_checkpoint in wav2vec2 config"
            )

    def test_xeus_language_not_supported(self, wav2vec2_router: LanguageRouter) -> None:
        with pytest.raises(LanguageNotSupportedError):
            wav2vec2_router.resolve("hu")


# ── Cross-config consistency ──────────────────────────────────────────────────


class TestCrossConfigConsistency:
    def test_merged_checkpoint_path_identical_across_configs(
        self,
        demo_router: LanguageRouter,
        merged_router: LanguageRouter,
    ) -> None:
        """Both configs must reference the same merged checkpoint file."""
        demo_merged_path = demo_router.resolve("es").checkpoint_path
        only_merged_path = merged_router.resolve("es").checkpoint_path
        assert demo_merged_path == only_merged_path

    def test_languages_in_merged_only_subset_of_demo(
        self,
        demo_router: LanguageRouter,
        merged_router: LanguageRouter,
    ) -> None:
        """Every language in merged-only config must also be in demo config."""
        demo_codes = set(demo_router.supported_languages())
        merged_codes = set(merged_router.supported_languages())
        extra = merged_codes - demo_codes
        assert not extra, f"merged-only has languages not in demo config: {extra}"

    def test_wav2vec2_languages_also_in_demo(
        self,
        demo_router: LanguageRouter,
        wav2vec2_router: LanguageRouter,
    ) -> None:
        demo_codes = set(demo_router.supported_languages())
        for code in wav2vec2_router.supported_languages():
            assert code in demo_codes, f"{code} in wav2vec2 config but not in demo"
