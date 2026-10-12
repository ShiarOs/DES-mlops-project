from psycopg_pool import ConnectionPool


def create_prediction_table(pool: ConnectionPool) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                pickup_location_id TEXT NOT NULL,
                dropoff_location_id TEXT NOT NULL,
                trip_distance DOUBLE PRECISION NOT NULL,
                predicted_duration_minutes DOUBLE PRECISION NOT NULL
            )
            """
        )


def save_prediction(
    pool: ConnectionPool,
    pickup_location_id: int,
    dropoff_location_id: int,
    trip_distance: float,
    predicted_duration_minutes: float,
) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_logs (
                pickup_location_id,
                dropoff_location_id,
                trip_distance,
                predicted_duration_minutes
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                str(pickup_location_id),
                str(dropoff_location_id),
                trip_distance,
                predicted_duration_minutes,
            ),
        )