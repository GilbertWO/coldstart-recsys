package com.coldstart.recsapi;

import java.util.List;

public record StoredList(List<Integer> items, String modelVersion) {}