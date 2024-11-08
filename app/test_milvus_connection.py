from pymilvus import connections


def test_connection():
    uri = "sqlite:///app/milvus_demo.db"
    try:
        connections.connect(alias="test", uri=uri)
        print("Successfully connected to Milvus Lite.")
    except Exception as e:
        print(f"Failed to connect to Milvus Lite: {e}")


if __name__ == "__main__":
    test_connection()
