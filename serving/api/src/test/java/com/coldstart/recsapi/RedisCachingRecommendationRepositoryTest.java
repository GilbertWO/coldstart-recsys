package com.coldstart.recsapi;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.junit.jupiter.api.Test;

class RedisCachingRecommendationRepositoryTest {

    private static final Duration TTL = Duration.ofHours(1);
    private static final String ID = "a".repeat(64);
    private static final StoredList LIST =
            new StoredList(List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), "v1");

    @Test
    void find_miss_loadsFromTheDelegateAndStoresWithTheTtl() {
        FakeStore store = new FakeStore();
        CountingDelegate delegate = new CountingDelegate();
        var repo = new RedisCachingRecommendationRepository(delegate, store, TTL);

        Optional<StoredList> result = repo.find(ID, "repurchase");

        assertThat(result).contains(LIST);
        assertThat(delegate.findCalls).isEqualTo(1);
        assertThat(store.values).containsKey("rec:repurchase:" + ID);
        assertThat(store.lastTtl).isEqualTo(TTL);
    }

    @Test
    void find_hit_doesNotCallTheDelegateAndReturnsTheSameList() {
        FakeStore store = new FakeStore();
        CountingDelegate delegate = new CountingDelegate();
        var repo = new RedisCachingRecommendationRepository(delegate, store, TTL);

        repo.find(ID, "repurchase");
        Optional<StoredList> second = repo.find(ID, "repurchase");

        assertThat(second).contains(LIST);
        assertThat(delegate.findCalls).isEqualTo(1);
    }

    @Test
    void find_emptyFromTheDelegate_isNotCached() {
        FakeStore store = new FakeStore();
        CountingDelegate delegate = new CountingDelegate();
        delegate.result = Optional.empty();
        var repo = new RedisCachingRecommendationRepository(delegate, store, TTL);

        assertThat(repo.find(ID, "repurchase")).isEmpty();
        assertThat(repo.find(ID, "repurchase")).isEmpty();

        assertThat(delegate.findCalls).isEqualTo(2);
        assertThat(store.values).isEmpty();
    }

    @Test
    void find_cacheReadFailure_fallsThroughToTheDelegate() {
        FakeStore store = new FakeStore();
        store.failReads = true;
        CountingDelegate delegate = new CountingDelegate();
        var repo = new RedisCachingRecommendationRepository(delegate, store, TTL);

        assertThat(repo.find(ID, "repurchase")).contains(LIST);
        assertThat(delegate.findCalls).isEqualTo(1);
    }

    @Test
    void find_cacheWriteFailure_stillReturnsTheResult() {
        FakeStore store = new FakeStore();
        store.failWrites = true;
        CountingDelegate delegate = new CountingDelegate();
        var repo = new RedisCachingRecommendationRepository(delegate, store, TTL);

        assertThat(repo.find(ID, "repurchase")).contains(LIST);
    }

    @Test
    void find_corruptCachedValue_isTreatedAsAMissAndOverwritten() {
        FakeStore store = new FakeStore();
        store.values.put("rec:repurchase:" + ID, "garbage");
        CountingDelegate delegate = new CountingDelegate();
        var repo = new RedisCachingRecommendationRepository(delegate, store, TTL);

        assertThat(repo.find(ID, "repurchase")).contains(LIST);
        assertThat(repo.find(ID, "repurchase")).contains(LIST);

        assertThat(delegate.findCalls).isEqualTo(1);
    }

    @Test
    void find_differentModes_useDifferentKeys() {
        FakeStore store = new FakeStore();
        CountingDelegate delegate = new CountingDelegate();
        var repo = new RedisCachingRecommendationRepository(delegate, store, TTL);

        repo.find(ID, "repurchase");
        repo.find(ID, "discover");

        assertThat(delegate.findCalls).isEqualTo(2);
        assertThat(store.values).containsOnlyKeys(
                "rec:repurchase:" + ID, "rec:discover:" + ID);
    }

    @Test
    void latestFallback_passesThroughAndNeverTouchesTheStore() {
        FakeStore store = new FakeStore();
        CountingDelegate delegate = new CountingDelegate();
        delegate.fallback = Optional.of(LIST);
        var repo = new RedisCachingRecommendationRepository(delegate, store, TTL);

        assertThat(repo.latestFallback()).contains(LIST);
        assertThat(store.getCalls).isZero();
        assertThat(store.values).isEmpty();
    }

    @Test
    void find_cacheReadFailure_doesNotAttemptAWrite() {
        int[] puts = {0};
        CacheStore store = new CacheStore() {
            @Override
            public Optional<String> get(String key) {
                throw new IllegalStateException("redis down");
            }

            @Override
            public void put(String key, String value, Duration ttl) {
                puts[0]++;
            }
        };
        var repo = new RedisCachingRecommendationRepository(new CountingDelegate(), store, TTL);

        assertThat(repo.find(ID, "repurchase")).contains(LIST);
        assertThat(puts[0]).isZero();
    }

    static class FakeStore implements CacheStore {
        final Map<String, String> values = new HashMap<>();
        Duration lastTtl;
        boolean failReads;
        boolean failWrites;
        int getCalls;

        @Override
        public Optional<String> get(String key) {
            getCalls++;
            if (failReads) {
                throw new IllegalStateException("redis down");
            }
            return Optional.ofNullable(values.get(key));
        }

        @Override
        public void put(String key, String value, Duration ttl) {
            if (failWrites) {
                throw new IllegalStateException("redis down");
            }
            values.put(key, value);
            lastTtl = ttl;
        }
    }

    static class CountingDelegate implements RecommendationRepository {
        Optional<StoredList> result = Optional.of(LIST);
        Optional<StoredList> fallback = Optional.empty();
        int findCalls;

        @Override
        public Optional<StoredList> find(String customerId, String mode) {
            findCalls++;
            return result;
        }

        @Override
        public Optional<StoredList> latestFallback() {
            return fallback;
        }
    }
}
