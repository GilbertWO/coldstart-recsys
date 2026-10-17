package com.coldstart.recsapi;

import java.time.Clock;
import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.data.redis.core.StringRedisTemplate;

@Configuration
public class CacheConfig {

    @Bean
    public Clock clock() {
        return Clock.systemUTC();
    }

    @Bean
    public CacheStore cacheStore(StringRedisTemplate redis) {
        return new RedisCacheStore(redis);
    }

    @Bean
    @Primary
    public RecommendationRepository cachingRecommendationRepository(
            JdbcRecommendationRepository jdbc,
            CacheStore store,
            Clock clock,
            @Value("${recs.list-cache-enabled:false}") boolean listCacheEnabled,
            @Value("${recs.list-cache-ttl:PT1H}") Duration listTtl,
            @Value("${recs.fallback-cache-ttl:PT1M}") Duration fallbackTtl) {
        RecommendationRepository source = listCacheEnabled
                ? new RedisCachingRecommendationRepository(jdbc, store, listTtl)
                : jdbc;
        return new CachingRecommendationRepository(source, clock, fallbackTtl);
    }
}
