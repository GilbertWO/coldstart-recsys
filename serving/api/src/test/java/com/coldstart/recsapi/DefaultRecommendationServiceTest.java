package com.coldstart.recsapi;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DefaultRecommendationServiceTest {

    private static final String KNOWN = "a".repeat(64);
    private static final String NEW_USER = "b".repeat(64);
    private static final List<Integer> REPURCHASE = List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    private static final List<Integer> DISCOVER = List.of(101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112);
    private static final List<Integer> POPULAR = List.of(201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212);

    private final FakeRepository repo = new FakeRepository();
    private final RecommendationService service = new DefaultRecommendationService(repo);

    @Test
    void knownCustomer_returnsStoredListAsAls() {
        repo.put(KNOWN, "repurchase", new StoredList(REPURCHASE, "als-v1"));

        RecommendationResponse r = service.recommend(KNOWN, "repurchase");

        assertThat(r.customerId()).isEqualTo(KNOWN);
        assertThat(r.mode()).isEqualTo("repurchase");
        assertThat(r.source()).isEqualTo("als");
        assertThat(r.modelVersion()).isEqualTo("als-v1");
        assertThat(r.items()).isEqualTo(REPURCHASE);
    }

    @Test
    void knownCustomer_getsTheListForTheRequestedMode() {
        repo.put(KNOWN, "repurchase", new StoredList(REPURCHASE, "als-v1"));
        repo.put(KNOWN, "discover", new StoredList(DISCOVER, "als-v1"));

        assertThat(service.recommend(KNOWN, "discover").items()).isEqualTo(DISCOVER);
        assertThat(service.recommend(KNOWN, "repurchase").items()).isEqualTo(REPURCHASE);
    }

    @Test
    void unknownCustomer_getsFallbackAndEchoesRequestedMode() {
        repo.fallback = new StoredList(POPULAR, "als-v1");

        RecommendationResponse r = service.recommend(NEW_USER, "discover");

        assertThat(r.source()).isEqualTo("fallback_popularity_7d");
        assertThat(r.mode()).isEqualTo("discover");
        assertThat(r.items()).isEqualTo(POPULAR);
    }

    @Test
    void unknownCustomer_getsSameFallbackListInBothModes() {
        repo.fallback = new StoredList(POPULAR, "als-v1");

        assertThat(service.recommend(NEW_USER, "repurchase").items())
                .isEqualTo(service.recommend(NEW_USER, "discover").items());
    }

    @Test
    void unknownCustomer_withNoFallbackRow_throwsInsteadOfInventingData() {
        assertThatThrownBy(() -> service.recommend(NEW_USER, "repurchase"))
                .isInstanceOf(FallbackUnavailableException.class);
    }

    private static class FakeRepository implements RecommendationRepository {
        private final Map<String, StoredList> rows = new HashMap<>();
        StoredList fallback; // null means the fallback table is empty

        void put(String customerId, String mode, StoredList list) {
            rows.put(customerId + "|" + mode, list);
        }

        @Override
        public Optional<StoredList> find(String customerId, String mode) {
            return Optional.ofNullable(rows.get(customerId + "|" + mode));
        }

        @Override
        public Optional<StoredList> latestFallback() {
            return Optional.ofNullable(fallback);
        }
    }
}