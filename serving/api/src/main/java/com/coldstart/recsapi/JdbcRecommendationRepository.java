package com.coldstart.recsapi;

import java.sql.Array;
import java.sql.SQLException;
import java.util.List;
import java.util.Optional;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Repository;

@Repository
public class JdbcRecommendationRepository implements RecommendationRepository {

    private final JdbcClient jdbc;

    public JdbcRecommendationRepository(JdbcClient jdbc) {
        this.jdbc = jdbc;
    }

    @Override
    public Optional<StoredList> find(String customerId, String mode) {
        return jdbc.sql("""
                        SELECT items, model_version
                        FROM recommendations
                        WHERE customer_id = :customerId AND mode = :mode
                        """)
                .param("customerId", customerId)
                .param("mode", mode)
                .query((rs, rowNum) -> new StoredList(
                        toList(rs.getArray("items")), rs.getString("model_version")))
                .optional();
    }

    @Override
    public Optional<StoredList> latestFallback() {
        return jdbc.sql("""
                        SELECT items, model_version
                        FROM fallback_lists
                        WHERE source = :source
                        ORDER BY as_of DESC
                        LIMIT 1
                        """)
                .param("source", DefaultRecommendationService.SOURCE_FALLBACK)
                .query((rs, rowNum) -> new StoredList(
                        toList(rs.getArray("items")), rs.getString("model_version")))
                .optional();
    }

    private static List<Integer> toList(Array array) throws SQLException {
        try {
            return List.of((Integer[]) array.getArray());
        } finally {
            array.free();
        }
    }
}