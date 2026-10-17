package com.coldstart.recsapi;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;

@SpringBootTest(properties = "recs.list-cache-enabled=true")
class RedisHealthSwitchOnTest {

    @Autowired
    ApplicationContext context;

    @Test
    void whenTheLayerIsOn_redisIsPartOfTheHealthCheck() {
        assertThat(Arrays.stream(context.getBeanDefinitionNames())
                .filter(n -> n.toLowerCase().contains("redis") && n.toLowerCase().contains("health"))
                .toList()).isNotEmpty();
    }
}
