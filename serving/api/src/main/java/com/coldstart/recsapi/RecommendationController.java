package com.coldstart.recsapi;

import jakarta.validation.constraints.Pattern;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class RecommendationController {

    private final RecommendationService service;

    public RecommendationController(RecommendationService service) {
        this.service = service;
    }

    @GetMapping("/v1/recommendations/{customerId}")
    public RecommendationResponse recommendations(
            @PathVariable @Pattern(regexp = "[0-9a-f]{64}") String customerId,
            @RequestParam(defaultValue = "repurchase")
            @Pattern(regexp = "repurchase|discover") String mode) {
        return service.recommend(customerId, mode);
    }
}