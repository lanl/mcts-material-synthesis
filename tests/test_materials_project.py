"""Tests for Materials Project integration."""

import pytest
from synthesis_planner.materials_project import MaterialsProjectClient, create_mp_client_from_config


def test_create_mp_client_disabled():
    """Test that client is not created when disabled"""
    config = {"materials_project": {"enable": False}}
    client = create_mp_client_from_config(config)
    assert client is None


def test_create_mp_client_no_api_key():
    """Test that client is not created without API key"""
    config = {"materials_project": {"enable": True, "api_key": ""}}
    client = create_mp_client_from_config(config)
    assert client is None


def test_create_mp_client_with_key():
    """Test that client is created with API key"""
    config = {"materials_project": {"enable": True, "api_key": "test-key"}}
    client = create_mp_client_from_config(config)
    assert client is not None
    assert isinstance(client, MaterialsProjectClient)


def test_mp_client_handles_missing_module():
    """Test that client handles missing mp-api gracefully"""
    # This will fail to import but shouldn't crash
    client = MaterialsProjectClient("test-key")
    # Should either be enabled (if mp-api installed) or disabled (if not)
    assert isinstance(client.enabled, bool)


def test_mp_client_get_thermodynamic_data_disabled():
    """Test that disabled client returns None"""
    client = MaterialsProjectClient("")
    client.enabled = False
    data = client.get_thermodynamic_data("BaTiO3")
    assert data is None


@pytest.mark.skip(reason="Requires valid Materials Project API key")
def test_mp_client_real_query():
    """
    Integration test with real MP API.
    Skipped by default - requires valid API key in config.py
    """
    import os
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        pytest.skip("MP_API_KEY not set")

    client = MaterialsProjectClient(api_key)
    if not client.enabled:
        pytest.skip("mp-api not available")

    # Test with known stable compound
    data = client.get_thermodynamic_data("BaTiO3")
    assert data is not None
    assert data.formula == "BaTiO3"
    # BaTiO3 is thermodynamically stable
    assert data.hull_energy_ev_per_atom is not None
    assert data.hull_energy_ev_per_atom < 0.05  # Should be stable or very close
