from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.task_models import ModelInfo
from app.repositories.task_repository import ModelInfoRepository


def test_get_model_info_resolves_short_model_alias_from_model_name(tmp_path) -> None:
    db_path = tmp_path / "model-info.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    session = Session()
    try:
        session.add(
            ModelInfo(
                model_id="a0d41440-c5c8-4756-a8c3-a5efb62ef327",
                model_name="hermes-bank-core3-2024-1775964140",
                model_type="lightgbm",
                version="1.0.0",
                file_path="/tmp/fake-model.pkl",
                status="ready",
            )
        )
        session.commit()

        repository = ModelInfoRepository(session)
        model_info = repository.get_model_info("bank-core3")

        assert model_info is not None
        assert model_info.model_id == "a0d41440-c5c8-4756-a8c3-a5efb62ef327"
        assert model_info.model_name == "hermes-bank-core3-2024-1775964140"
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
