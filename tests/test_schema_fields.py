import pytest

from app.utils import calculate_field_accuracy


def test_calculate_field_accuracy():
    # Mock input fields that should be 100% correct
    test_fields = {
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 25,
        "objective_end_time": "datetime.datetime(2024, 5, 21, 22, 30, 0, 250000, tzinfo=TzInfo(UTC))",
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, 0, 150000, tzinfo=TzInfo(UTC))",
        "orbital_regime": "LEO",
        "patience_minutes": 10,
        "priority": 12,
        "rso_id_list": [],
        "sensor_name_list": ["RME02", "LMNT01"],
    }

    # Test perfect match
    accuracy, correct, total = calculate_field_accuracy(test_fields)
    assert accuracy == 100.0
    assert correct == 11
    assert total == 11

    # Test with one wrong field
    test_fields_with_error = test_fields.copy()
    test_fields_with_error["priority"] = 999
    accuracy, correct, total = calculate_field_accuracy(test_fields_with_error)
    assert accuracy == pytest.approx(90.9, 0.1)  # 10/11 ≈ 90.9%
    assert correct == 10
    assert total == 11

    # Test with wrong datetime format but same time
    test_fields_with_different_datetime = test_fields.copy()
    test_fields_with_different_datetime["objective_end_time"] = (
        "datetime.datetime(2024, 5, 21, 22, 30, 0, 250000)"
    )
    accuracy, correct, total = calculate_field_accuracy(
        test_fields_with_different_datetime
    )
    assert (
        accuracy == 100.0
    )  # Should still be 100% as we're parsing datetime strings flexibly

    # Test with different sensor list order
    test_fields_with_different_order = test_fields.copy()
    test_fields_with_different_order["sensor_name_list"] = [
        "LMNT01",
        "RME02",
    ]  # Reversed order
    accuracy, correct, total = calculate_field_accuracy(
        test_fields_with_different_order
    )
    assert accuracy == 100.0  # Should be 100% as we're sorting lists before comparison


def test_missing_fields():
    # Test with missing field
    incomplete_fields = {
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        # missing some fields
    }
    accuracy, correct, total = calculate_field_accuracy(incomplete_fields)
    assert accuracy < 100.0
    assert correct < total


def test_edge_cases():
    # Empty fields
    empty_fields = {}
    accuracy, correct, total = calculate_field_accuracy(empty_fields)
    assert accuracy == 0.0
    assert correct == 0
    assert total > 0


def test_datetime_field_accuracy():
    # Base correct fields with focus on datetime
    base_fields = {
        "objective_end_time": "datetime.datetime(2024, 5, 21, 22, 30, 0, 250000, tzinfo=TzInfo(UTC))",
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, 0, 150000, tzinfo=TzInfo(UTC))",
        # Minimum other fields needed
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 25,
        "orbital_regime": "LEO",
        "patience_minutes": 10,
        "priority": 12,
        "rso_id_list": [],
        "sensor_name_list": ["RME02", "LMNT01"],
    }

    # Test 1: Original format
    accuracy, correct, total = calculate_field_accuracy(base_fields)
    assert accuracy == 100.0, "Base datetime format should match"

    # Test 2: Alternative datetime string formats
    variant_formats = {
        **base_fields,
        "objective_end_time": "2024-05-21 22:30:00.250000 UTC",
        "objective_start_time": "2024-05-21 19:20:00.150000 UTC",
    }
    accuracy, correct, total = calculate_field_accuracy(variant_formats)
    assert accuracy == 100.0, "Alternative datetime format should match"

    # Test 3: With different timezone formats
    tz_variants = {
        **base_fields,
        "objective_end_time": "datetime.datetime(2024, 5, 21, 22, 30, 0, 250000, tzinfo=timezone.utc)",
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, 0, 150000, tzinfo=timezone.utc)",
    }
    accuracy, correct, total = calculate_field_accuracy(tz_variants)
    assert accuracy == 100.0, "Different timezone format should match"

    # Test 4: Without microseconds
    no_microseconds = {
        **base_fields,
        "objective_end_time": "datetime.datetime(2024, 5, 21, 22, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, 0, tzinfo=TzInfo(UTC))",
    }
    accuracy, correct, total = calculate_field_accuracy(no_microseconds)
    assert accuracy == 100.0, "Datetime without microseconds should match"

    # Test 5: Wrong times
    wrong_times = {
        **base_fields,
        "objective_end_time": "datetime.datetime(2024, 5, 21, 23, 30, 0, 250000, tzinfo=TzInfo(UTC))",  # Wrong hour
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, 0, 150000, tzinfo=TzInfo(UTC))",
    }
    accuracy, correct, total = calculate_field_accuracy(wrong_times)
    assert accuracy < 100.0, "Wrong time should not match"

    # Test 6: ISO format
    iso_format = {
        **base_fields,
        "objective_end_time": "2024-05-21T22:30:00.250000+00:00",
        "objective_start_time": "2024-05-21T19:20:00.150000+00:00",
    }
    accuracy, correct, total = calculate_field_accuracy(iso_format)
    assert accuracy == 100.0, "ISO format should match"

    # Test 7: Different but equivalent times (UTC vs +00:00)
    equivalent_times = {
        **base_fields,
        "objective_end_time": "2024-05-21T22:30:00.250000+00:00",
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, 0, 150000, tzinfo=TzInfo(UTC))",
    }
    accuracy, correct, total = calculate_field_accuracy(equivalent_times)
    assert accuracy == 100.0, "Equivalent times in different formats should match"
