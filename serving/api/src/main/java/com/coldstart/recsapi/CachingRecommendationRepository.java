package com.coldstart.recsapi;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

public class CachingRecommendationRepository implements RecommendationRepository {

    private record Entry(StoredList value, Instant expiresAt) {}

    private final RecommendationRepository delegate;
    private final Clock clock;
    private final Duration ttl;
    private final AtomicReference<Entry> fallback = new AtomicReference<>();

    public CachingRecommendationRepository(RecommendationRepository delegate, Clock clock, Duration ttl) {
        this.delegate = delegate;
        this.clock = clock;
        this.ttl = ttl;
    }

    @Override
    public Optional<StoredList> find(String customerId, String mode) {
        return delegate.find(customerId, mode);
    }

    @Override
    public Optional<StoredList> latestFallback() {
        Instant now = clock.instant();
        Entry cached = fallback.get();
        if (cached != null && now.isBefore(cached.expiresAt())) {
            return Optional.of(cached.value());
        }
        Optional<StoredList> loaded = delegate.latestFallback();
        loaded.ifPresent(value -> fallback.set(new Entry(value, now.plus(ttl))));
        return loaded;
    }
}