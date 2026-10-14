package com.coldstart.recsapi;

import java.time.Duration;
import java.util.Optional;

public interface CacheStore {

    Optional<String> get(String key);

    void put(String key, String value, Duration ttl);
}
