package com.coldstart.recsapi;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RedisCachingRecommendationRepository implements RecommendationRepository {

    private static final Logger log =
            LoggerFactory.getLogger(RedisCachingRecommendationRepository.class);

    private final RecommendationRepository delegate;
    private final CacheStore store;
    private final Duration ttl;

    public RedisCachingRecommendationRepository(
            RecommendationRepository delegate, CacheStore store, Duration ttl) {
        this.delegate = delegate;
        this.store = store;
        this.ttl = ttl;
    }

    @Override
    public Optional<StoredList> find(String customerId, String mode) {
        String key = "rec:" + mode + ":" + customerId;
        Optional<String> raw;
        try {
            raw = store.get(key);
        } catch (RuntimeException e) {
            log.warn("cache read failed, skipping the cache for this request: {}", e.toString());
            return delegate.find(customerId, mode);
        }
        Optional<StoredList> cached = raw.flatMap(RedisCachingRecommendationRepository::decode);
        if (cached.isPresent()) {
            return cached;
        }
        Optional<StoredList> loaded = delegate.find(customerId, mode);
        loaded.ifPresent(value -> write(key, value));
        return loaded;
    }

    @Override
    public Optional<StoredList> latestFallback() {
        return delegate.latestFallback();
    }

    private void write(String key, StoredList value) {
        try {
            store.put(key, encode(value), ttl);
        } catch (RuntimeException e) {
            log.warn("cache write failed: {}", e.toString());
        }
    }

    private static String encode(StoredList value) {
        String ids = value.items().stream()
                .map(String::valueOf)
                .collect(Collectors.joining(","));
        return value.modelVersion() + "|" + ids;
    }

    static Optional<StoredList> decode(String raw) {
        try {
            int bar = raw.indexOf('|');
            if (bar <= 0) {
                return Optional.empty();
            }
            String version = raw.substring(0, bar);
            String[] parts = raw.substring(bar + 1).split(",");
            List<Integer> items = new ArrayList<>(parts.length);
            for (String part : parts) {
                items.add(Integer.parseInt(part));
            }
            return Optional.of(new StoredList(List.copyOf(items), version));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }
}
