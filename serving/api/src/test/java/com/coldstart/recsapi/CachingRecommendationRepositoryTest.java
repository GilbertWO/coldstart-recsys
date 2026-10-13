package com.coldstart.recsapi;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class CachingRecommendationRepositoryTest {

    private static final Duration TTL = Duration.ofSeconds(60);
    private static final StoredList OLD =
            new StoredList(List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), "v1");
    private static final StoredList NEW =
            new StoredList(List.of(101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112), "v2");

    private final MutableClock clock = new MutableClock();
    private final CountingDelegate delegate = new CountingDelegate();
    private final CachingRecommendationRepository repo =
            new CachingRecommendationRepository(delegate, clock, TTL);

    @Test
    void fallback_secondCallWithinTtl_doesNotHitTheDelegate() {
        delegate.fallback = OLD;

        assertThat(repo.latestFallback()).contains(OLD);
        clock.advance(Duration.ofSeconds(59));
        assertThat(repo.latestFallback()).contains(OLD);

        assertThat(delegate.fallbackCalls).isEqualTo(1);
    }

    @Test
    void fallback_afterTtl_reloadsAndPicksUpTheNewList() {
        delegate.fallback = OLD;
        repo.latestFallback();

        delegate.fallback = NEW;
        clock.advance(TTL.plusSeconds(1));

        assertThat(repo.latestFallback()).contains(NEW);
        assertThat(delegate.fallbackCalls).isEqualTo(2);
    }

    @Test
    void fallback_emptyResultIsNotCached() {
        delegate.fallback = null; // fallback table empty
        assertThat(repo.latestFallback()).isEmpty();

        delegate.fallback = OLD;  // table gets filled a moment later
        assertThat(repo.latestFallback()).contains(OLD);
    }

    @Test
    void find_isNeverCached() {
        repo.find("a".repeat(64), "repurchase");
        repo.find("a".repeat(64), "repurchase");

        assertThat(delegate.findCalls).isEqualTo(2);
    }

    private static class CountingDelegate implements RecommendationRepository {
        StoredList fallback;
        int fallbackCalls;
        int findCalls;

        @Override
        public Optional<StoredList> find(String customerId, String mode) {
            findCalls++;
            return Optional.of(OLD);
        }

        @Override
        public Optional<StoredList> latestFallback() {
            fallbackCalls++;
            return Optional.ofNullable(fallback);
        }
    }

    private static class MutableClock extends Clock {
        private Instant now = Instant.parse("2026-10-13T00:00:00Z");

        void advance(Duration d) {
            now = now.plus(d);
        }

        @Override public ZoneId getZone() { return ZoneOffset.UTC; }
        @Override public Clock withZone(ZoneId zone) { return this; }
        @Override public Instant instant() { return now; }
    }
}