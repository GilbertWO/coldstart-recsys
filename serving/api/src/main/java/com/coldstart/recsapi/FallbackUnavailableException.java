package com.coldstart.recsapi;

public class FallbackUnavailableException extends RuntimeException {
    public FallbackUnavailableException(String message) {
        super(message);
    }
}