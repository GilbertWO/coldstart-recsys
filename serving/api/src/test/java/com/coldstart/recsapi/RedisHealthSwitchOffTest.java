package com.coldstart.recsapi;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;

@SpringBootTest
class RedisHealthSwitchOffTest {

    @Autowired
    ApplicationContext context;

    @Test
    void whenTheLayerIsOff_redisIsNotPartOfTheHealthCheck() {
        assertThat(Arrays.stream(context.getBeanDefinitionNames())
                .filter(n -> n.toLowerCase().contains("redis") && n.toLowerCase().contains("health"))
                .toList()).isEmpty();
    }
}
