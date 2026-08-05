from __future__ import annotations

from pathlib import Path

from helion_rag.setup_helpers import write_environment


def test_generated_environment_contains_keys_and_vertex_without_api_key(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".helion-rag"
    path = root / "env.sh"

    write_environment(
        path,
        rag_root=root,
        hardware_family="h100",
        manifold_base="manifold://artifacts",
        embed_model="Qwen/Qwen3-Embedding-4B",
        private_key_path=root / "publisher-private.pem",
        public_key_path=root / "publisher-public.pem",
        llm_provider="vertex",
        llm_model="claude-opus-4-8",
        llm_ca_bundle="/etc/company-ca.pem",
        vertex_base_url="https://vertex.example",
        vertex_project_id="project",
        cloud_ml_region="us-west1",
        hf_home=tmp_path / "hf cache",
        collect_autotune=True,
    )

    text = path.read_text(encoding="utf-8")
    assert "export HELION_RAG_PRIVATE_KEY_PATH=" in text
    assert "export HELION_RAG_PUBLIC_KEY_PATH=" in text
    assert "export HELION_LLM_PROVIDER='vertex'" in text
    assert "export HELION_LLM_MODEL='claude-opus-4-8'" in text
    assert "export HELION_LLM_CA_BUNDLE='/etc/company-ca.pem'" in text
    assert "export ANTHROPIC_VERTEX_BASE_URL='https://vertex.example'" in text
    assert "export ANTHROPIC_VERTEX_PROJECT_ID='project'" in text
    assert "export CLOUD_ML_REGION='us-west1'" in text
    assert "export HF_HOME=" in text
    assert "HELION_AUTOTUNE_LOG_DETAILS='1'" in text
    assert "API_KEY" not in text


def test_generated_environment_omits_unset_optional_vertex_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rag"
    path = root / "env.sh"
    write_environment(
        path,
        rag_root=root,
        hardware_family="h100",
        manifold_base="m",
        embed_model="qwen",
        private_key_path=root / "private.pem",
        public_key_path=root / "public.pem",
        llm_provider="vertex",
        llm_model="model",
        llm_ca_bundle="/ca.pem",
    )

    text = path.read_text(encoding="utf-8")
    assert "ANTHROPIC_VERTEX_BASE_URL" not in text
    assert "ANTHROPIC_VERTEX_PROJECT_ID" not in text
    assert "CLOUD_ML_REGION" not in text
