package com.coldstart.recsapi;

import java.util.List;

public record RecommendationResponse(
        String customerId, String mode, String source,
        String modelVersion, List<Integer> items) {}