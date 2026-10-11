package com.coldstart.recsapi;

import org.springframework.stereotype.Service;

@Service
public class DefaultRecommendationService implements RecommendationService {

    static final String SOURCE_ALS = "als";
    static final String SOURCE_FALLBACK = "fallback_popularity_7d";

    private final RecommendationRepository repository;

    public DefaultRecommendationService(RecommendationRepository repository) {
        this.repository = repository;
    }

    @Override
    public RecommendationResponse recommend(String customerId, String mode) {
        return repository.find(customerId, mode)
                .map(stored -> response(customerId, mode, SOURCE_ALS, stored))
                .orElseGet(() -> fallback(customerId, mode));
    }

    private RecommendationResponse fallback(String customerId, String mode) {
        StoredList stored = repository.latestFallback()
                .orElseThrow(() -> new FallbackUnavailableException(
                        "no fallback list available in fallback_lists"));
        return response(customerId, mode, SOURCE_FALLBACK, stored);
    }

    private static RecommendationResponse response(
            String customerId, String mode, String source, StoredList stored) {
        return new RecommendationResponse(
                customerId, mode, source, stored.modelVersion(), stored.items());
    }
}