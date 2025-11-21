import os
import json
import time
import pytest
from django.urls import reverse
from django.conf import settings


@pytest.mark.django_db
def test_start_training(client, monkeypatch):
    """
    Test bahwa endpoint start_training bisa dipanggil dan
    mengubah status menjadi 'training'.
    """

    # Mock threading supaya tidak benar-benar training
    class DummyThread:
        def __init__(*args, **kwargs): pass
        def start(*args, **kwargs): pass

    monkeypatch.setattr("training_api.threading.Thread", DummyThread)

    url = reverse("start_training")
    response = client.post(url)

    assert response.status_code == 200
    data = response.json()
    assert "Training dimulai!" in data["message"]


@pytest.mark.django_db
def test_stop_training(client):
    url = reverse("stop_training")
    response = client.post(url)

    assert response.status_code == 200
    assert "Stop training dikirim!" in response.json()["message"]


@pytest.mark.django_db
def test_get_progress(client):
    """
    Testing bahwa progress bisa dipanggil dan return JSON lengkap.
    """

    url = reverse("get_progress")
    response = client.get(url)

    assert response.status_code == 200
    data = response.json()

    # Pastikan field utama ada
    assert "epoch" in data
    assert "status" in data
    assert "metrics" in data
    assert "scatter_features" in data
    assert isinstance(data["progress"], int)


@pytest.mark.django_db
def test_download_model_cnn(client, tmp_path, monkeypatch):
    """
    Testing download model CNN
    """

    # Create dummy file
    dummy_model_path = tmp_path / "cnn_model.h5"
    dummy_model_path.write_text("dummy model")

    # Mock CNN path
    monkeypatch.setattr("training_api.cnn_path", str(dummy_model_path))

    url = reverse("download_model", kwargs={"model_type": "cnn"})
    response = client.get(url)

    assert response.status_code == 200
    assert response.get("Content-Disposition") is not None


@pytest.mark.django_db
def test_download_model_xgb(client, tmp_path, monkeypatch):
    """
    Testing download model XGB
    """

    dummy_model_path = tmp_path / "xgb_model.pkl"
    dummy_model_path.write_text("dummy model")

    monkeypatch.setattr("training_api.xgb_path", str(dummy_model_path))

    url = reverse("download_model", kwargs={"model_type": "xgb"})
    response = client.get(url)

    assert response.status_code == 200
    assert response.get("Content-Disposition") is not None


@pytest.mark.django_db
def test_download_model_invalid(client):
    url = reverse("download_model", kwargs={"model_type": "invalid"})
    response = client.get(url)

    assert response.status_code == 200
    assert "error" in response.json()
