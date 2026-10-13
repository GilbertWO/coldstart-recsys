package com.coldstart.recsapi;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class RepositoryWiringTest {

    @Autowired
    RecommendationRepository repository;

    @Test
    void theServiceGetsTheCachingRepository() {
        assertThat(repository).isInstanceOf(CachingRecommendationRepository.class);
    }
}