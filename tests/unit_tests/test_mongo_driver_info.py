"""Unit tests for MongoDB driver handshake metadata in MongoStorage."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_driver_info_name() -> None:
    """_DRIVER_INFO has name 'Upsonic'."""
    from upsonic.storage.mongo.mongo import _DRIVER_INFO

    assert _DRIVER_INFO is not None
    assert _DRIVER_INFO.name == "Upsonic"


def test_mongo_storage_passes_driver_info_to_client() -> None:
    """MongoStorage passes driver=_DRIVER_INFO when constructing MongoClient from db_url."""
    from upsonic.storage.mongo.mongo import _DRIVER_INFO

    with patch("upsonic.storage.mongo.mongo.MongoClient") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from upsonic.storage.mongo.mongo import MongoStorage

        MongoStorage(db_url="mongodb://localhost:27017", db_name="test_db")

        mock_client_cls.assert_called_once()
        _, kwargs = mock_client_cls.call_args
        assert kwargs.get("driver") is _DRIVER_INFO


def test_mongo_storage_does_not_override_provided_client() -> None:
    """When a db_client is supplied, MongoStorage does not call MongoClient."""
    with patch("upsonic.storage.mongo.mongo.MongoClient") as mock_client_cls:
        from upsonic.storage.mongo.mongo import MongoStorage

        provided_client = MagicMock()
        MongoStorage(db_client=provided_client, db_name="test_db")

        mock_client_cls.assert_not_called()
