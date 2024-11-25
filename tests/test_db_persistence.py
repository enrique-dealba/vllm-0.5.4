import subprocess
import time
import uuid

import pytest

# Constants
SERVICE_NAME = "timescaledb"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "password"
MAX_ATTEMPTS = 15
SLEEP_INTERVAL = 5  # seconds


@pytest.fixture(scope="session")
def get_timescaledb_container_id():
    """Retrieves the container ID for the timescaledb service using docker compose."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q", SERVICE_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        container_id = result.stdout.strip()
        assert container_id, f"No container ID found for service '{SERVICE_NAME}'"
        return container_id
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Failed to get container ID for service '{SERVICE_NAME}': {e.stderr}"
        )


@pytest.fixture(scope="session")
def insert_test_record(get_timescaledb_container_id):
    """Inserts a unique test record into the embeddings table.
    Returns the content value used for insertion.
    """
    unique_content = f"Test persistence {uuid.uuid4()}"
    insert_command = (
        f"INSERT INTO embeddings (id, metadata, content, embedding) "
        f"VALUES (gen_random_uuid(), '{{\"source\": \"test\"}}', '{unique_content}', array_fill(0.1, ARRAY[384]));"
    )
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                get_timescaledb_container_id,
                "psql",
                "-U",
                DB_USER,
                "-d",
                DB_NAME,
                "-c",
                insert_command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        assert "INSERT 0 1" in result.stdout, f"Unexpected psql output: {result.stdout}"
        return unique_content
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to insert test record: {e.stderr}")


def get_record_count(container_id, content):
    """Retrieves the count of records matching the given content."""
    select_command = f"SELECT COUNT(*) FROM embeddings WHERE content = '{content}';"
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container_id,
                "psql",
                "-U",
                DB_USER,
                "-d",
                DB_NAME,
                "-c",
                select_command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        # Parse the output
        # Expected psql output:
        #  count
        # -------
        #      1
        # (1 row)
        lines = result.stdout.strip().splitlines()
        assert len(lines) >= 3, f"Unexpected psql output: {result.stdout}"
        count_str = lines[2].strip()
        count = int(count_str)
        return count
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to query test record: {e.stderr}")
    except (IndexError, ValueError) as e:
        pytest.fail(f"Failed to parse query result: {e}")


def restart_service(service_name):
    """Restarts a Docker Compose service."""
    try:
        result = subprocess.run(
            ["docker", "compose", "restart", service_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to restart service '{service_name}': {e.stderr}")


def wait_for_health(
    container_id, max_attempts=MAX_ATTEMPTS, sleep_interval=SLEEP_INTERVAL
):
    """Waits until the Docker container is healthy, or until max_attempts is reached."""
    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Health.Status}}", container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            status = result.stdout.strip()
            if status == "healthy":
                print(f"Container '{container_id}' is healthy.")
                return
            elif status == "unhealthy":
                pytest.fail(f"Container '{container_id}' is unhealthy.")
            else:
                print(
                    f"Attempt {attempt}/{max_attempts}: Container health status: {status}"
                )
        except subprocess.CalledProcessError as e:
            print(
                f"Attempt {attempt}/{max_attempts}: Failed to get health status: {e.stderr}"
            )
        time.sleep(sleep_interval)
    pytest.fail(
        f"Container '{container_id}' did not become healthy within {max_attempts * sleep_interval} seconds."
    )


def get_container_id_from_compose(service_name):
    """Retrieves the container ID for a given service using docker compose."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q", service_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        container_id = result.stdout.strip()
        assert container_id, f"No container ID found for service '{service_name}'"
        return container_id
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Failed to get container ID for service '{service_name}': {e.stderr}"
        )


def test_database_persistence(get_timescaledb_container_id, insert_test_record):
    """Tests that data persists in the database after restarting the timescaledb container."""
    container_id = get_timescaledb_container_id
    content = insert_test_record

    # Verify the record exists before restart
    count_before = get_record_count(container_id, content)
    assert count_before == 1, f"Expected 1 record before restart, found {count_before}"
    print(f"Record count before restart: {count_before}")

    # Restart the timescaledb service
    print(f"Restarting service '{SERVICE_NAME}'...")
    restart_service(SERVICE_NAME)

    # Get the container ID again (in case it changed)
    new_container_id = get_container_id_from_compose(SERVICE_NAME)
    assert new_container_id == container_id, "Container ID changed after restart."

    # Wait for the container to become healthy
    print(f"Waiting for container '{new_container_id}' to become healthy...")
    wait_for_health(new_container_id)

    # Verify the record still exists after restart
    count_after = get_record_count(new_container_id, content)
    assert count_after == 1, f"Expected 1 record after restart, found {count_after}"
    print(f"Record count after restart: {count_after}")
