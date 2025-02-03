from app.config import settings
from app.utils import load_schema


def test_schema_update_propagation():
    # Initial schema
    settings.update_schema("ObjectiveType")
    initial_schema = load_schema()

    # Update schema
    settings.update_schema("CatalogMaintenanceObjective")
    updated_schema = load_schema()

    # Verify schemas are different
    assert initial_schema != updated_schema
    assert initial_schema.__name__ == "ObjectiveType"
    assert updated_schema.__name__ == "CatalogMaintenanceObjective"

    # Verify schema fields
    assert hasattr(updated_schema, "classification_marking")
    assert hasattr(updated_schema, "orbital_regime")

    # Reset schema
    settings.update_schema("ObjectiveType")
