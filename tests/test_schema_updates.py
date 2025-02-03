from app.config import settings
from app.utils import debug_schema, load_schema


def test_schema_update_propagation():
    # Initial schema
    settings.update_schema("ObjectiveType")
    initial_schema = load_schema()
    debug_schema(initial_schema)  # Debug initial schema

    # Update schema
    settings.update_schema("CatalogMaintenanceObjective")
    updated_schema = load_schema()
    debug_schema(updated_schema)  # Debug updated schema

    # Verify schemas are different
    assert initial_schema != updated_schema
    assert initial_schema.__name__ == "ObjectiveType"
    assert updated_schema.__name__ == "CatalogMaintenanceObjective"

    # Create an instance to verify field access
    test_instance = updated_schema(
        classification_marking="U",
        orbital_regime="LEO",
        data_mode="REAL",
        collect_request_type="RATE_TRACK_SIDEREAL",
        patience_minutes=30,
        end_time_offset_minutes=20,
        priority=1000,
        sensor_name_list=["RME02", "LMNT01"],
        objective_start_time="2024-05-21T19:20:00.150000+00:00",
        objective_end_time="2024-05-21T22:30:00.250000+00:00",
    )

    # Verify schema fields using instance
    assert hasattr(test_instance, "classification_marking")
    assert hasattr(test_instance, "orbital_regime")

    # Reset schema
    settings.update_schema("ObjectiveType")
