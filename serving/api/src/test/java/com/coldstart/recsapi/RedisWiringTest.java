package com.coldstart.recsapi;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.time.Duration;
import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;

@SpringBootTest(properties = "recs.list-cache-enabled=true")
class RedisWiringTest {

    private static final String ID = "b".repeat(64);

    @MockitoBean
    JdbcRecommendationRepository jdbc;

    @MockitoBean
    CacheStore store;

    @Autowired
    RecommendationRepository repository;

    @Test
    void aLookupGoesThroughTheRedisLayer() {
        StoredList list = new StoredList(List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), "v1");
        when(jdbc.find(ID, "repurchase")).thenReturn(Optional.of(list));

        assertThat(repository.find(ID, "repurchase")).contains(list);

        verify(store).get("rec:repurchase:" + ID);
        verify(store).put(eq("rec:repurchase:" + ID), anyString(), eq(Duration.ofHours(1)));
    }
}
