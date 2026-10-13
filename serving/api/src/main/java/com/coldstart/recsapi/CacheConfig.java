package com.coldstart.recsapi;

import java.time.Clock;
import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

@Configuration
public class CacheConfig {

    @Bean
    public Clock clock() {
        return Clock.systemUTC();
    }

    @Bean
    @Primary
    public RecommendationRepository cachingRecommendationRepository(
            JdbcRecommendationRepository jdbc,
            Clock clock,
            @Value("${recs.fallback-cache-ttl:PT1M}") Duration ttl) {
        return new CachingRecommendationRepository(jdbc, clock, ttl);
    }
}