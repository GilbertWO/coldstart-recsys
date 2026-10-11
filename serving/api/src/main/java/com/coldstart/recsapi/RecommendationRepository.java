package com.coldstart.recsapi;

import java.util.Optional;

public interface RecommendationRepository {
    Optional<StoredList> find(String customerId, String mode);
    Optional<StoredList> latestFallback();
}