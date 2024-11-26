import logging

import pandas as pd
import streamlit as st

from app.backend.vector_store import VectorStore
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_db():
    """Initialize database connection"""
    try:
        vs = VectorStore()
        return vs
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        logger.error(f"Database connection error: {e}", exc_info=True)
        return None


def get_db_stats(vector_store):
    """Get basic database statistics"""
    try:
        with vector_store.conn.cursor() as cur:
            # Total number of entries
            cur.execute(f"SELECT COUNT(*) FROM {settings.VECTOR_STORE_TABLE_NAME}")
            total_count = cur.fetchone()[0]

            # Entries in last 24 hours
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM {settings.VECTOR_STORE_TABLE_NAME} 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            recent_count = cur.fetchone()[0]

            # Average embedding similarity (self-similarity as health check)
            cur.execute(f"""
                SELECT AVG(1 - (embedding <=> embedding)) as avg_similarity
                FROM {settings.VECTOR_STORE_TABLE_NAME}
                LIMIT 1000
            """)
            avg_similarity = cur.fetchone()[0]

            return {
                "total_entries": total_count,
                "recent_entries": recent_count,
                "avg_similarity": avg_similarity or 0.0,
            }
    except Exception as e:
        logger.error(f"Error fetching database stats: {e}", exc_info=True)
        return None


def main():
    st.set_page_config(page_title="TimescaleDB Monitor", page_icon="📊", layout="wide")

    st.title("📊 TimescaleDB Monitor")

    # Initialize database connection
    vector_store = initialize_db()
    if not vector_store:
        st.stop()

    # Database Stats Section
    st.header("Database Overview")
    stats = get_db_stats(vector_store)
    if stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", f"{stats['total_entries']:,}")
        with col2:
            st.metric("Entries (Last 24h)", f"{stats['recent_entries']:,}")
        with col3:
            st.metric("Avg Similarity Score", f"{stats['avg_similarity']:.4f}")

    # Query Section
    st.header("Data Explorer")

    tab1, tab2 = st.tabs(["🔍 Search", "📈 Time Analysis"])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter search terms or leave empty to see all",
            )
        with col2:
            limit = st.number_input(
                "Result Limit", min_value=1, max_value=100, value=10
            )

        if st.button("Search", key="search_button"):
            with st.spinner("Fetching results..."):
                results = vector_store.search(
                    search_query if search_query else "", limit=limit
                )

                if results.empty:
                    st.info("No results found")
                else:
                    st.success(f"Found {len(results)} results")

                    for _, row in results.iterrows():
                        with st.expander(
                            f"Entry: {row['id'][:8]}... (Similarity: {row['similarity']:.4f})"
                        ):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.text_area(
                                    "Content",
                                    row["content"],
                                    height=100,
                                    disabled=True,
                                    key=f"content_{row['id']}",  # Add unique key using the row ID
                                )

                            with col2:
                                st.json(row["metadata"])

                            st.text(f"Created: {row['created_at']}")

    with tab2:
        st.subheader("Time-based Analysis")

        # Time range selector
        time_range = st.selectbox(
            "Time Range", ["Last 24 Hours", "Last Week", "Last Month", "All Time"]
        )

        range_map = {
            "Last 24 Hours": "24 hours",
            "Last Week": "7 days",
            "Last Month": "30 days",
            "All Time": "100 years",  # Effectively all time
        }

        try:
            with vector_store.conn.cursor() as cur:
                # Get time-based entry counts
                cur.execute(f"""
                    SELECT 
                        date_trunc('hour', created_at) as time_bucket,
                        COUNT(*) as entry_count
                    FROM {settings.VECTOR_STORE_TABLE_NAME}
                    WHERE created_at > NOW() - INTERVAL '{range_map[time_range]}'
                    GROUP BY time_bucket
                    ORDER BY time_bucket DESC
                """)

                time_data = cur.fetchall()
                if time_data:
                    df = pd.DataFrame(time_data, columns=["time", "count"])
                    st.line_chart(df.set_index("time"))
                else:
                    st.info("No time-based data available")

        except Exception as e:
            st.error(f"Error analyzing time data: {e}")
            logger.error("Time analysis error", exc_info=True)

    # Advanced Query Section
    with st.expander("Advanced Queries"):
        st.subheader("Metadata Analysis")

        try:
            with vector_store.conn.cursor() as cur:
                # Get unique metadata keys
                cur.execute(f"""
                    SELECT DISTINCT jsonb_object_keys(metadata)
                    FROM {settings.VECTOR_STORE_TABLE_NAME}
                """)
                metadata_keys = [row[0] for row in cur.fetchall()]

                if metadata_keys:
                    selected_key = st.selectbox("Select Metadata Field", metadata_keys)

                    if selected_key:
                        cur.execute(
                            f"""
                            SELECT metadata->%s as value, COUNT(*) as count
                            FROM {settings.VECTOR_STORE_TABLE_NAME}
                            GROUP BY metadata->%s
                            ORDER BY count DESC
                            LIMIT 10
                        """,
                            (selected_key, selected_key),
                        )

                        results = cur.fetchall()
                        if results:
                            df = pd.DataFrame(results, columns=["value", "count"])
                            st.bar_chart(df.set_index("value"))
                else:
                    st.info("No metadata fields found")

        except Exception as e:
            st.error(f"Error analyzing metadata: {e}")
            logger.error("Metadata analysis error", exc_info=True)


if __name__ == "__main__":
    main()
