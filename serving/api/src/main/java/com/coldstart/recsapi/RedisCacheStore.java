package com.coldstart.recsapi;

import java.time.Duration;
import java.util.Optional;

import org.springframework.data.redis.core.StringRedisTemplate;

public class RedisCacheStore implements CacheStore {

    private final StringRedisTemplate redis;

    public RedisCacheStore(StringRedisTemplate redis) {
        this.redis = redis;
    }

    @Override
    public Optional<String> get(String key) {
        return Optional.ofNullable(redis.opsForValue().get(key));
    }

    @Override
    public void put(String key, String value, Duration ttl) {
        redis.opsForValue().set(key, value, ttl);
    }
}
