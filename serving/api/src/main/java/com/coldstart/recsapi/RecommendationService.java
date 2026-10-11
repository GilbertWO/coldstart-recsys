package com.coldstart.recsapi;

public interface RecommendationService {
    RecommendationResponse recommend(String customerId, String mode);
}