package com.coldstart.recsapi;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(RecommendationController.class)
class RecommendationControllerContractTest {

    private static final String URL = "/v1/recommendations/{id}";
    private static final String ID = "a".repeat(64); // valid format: 64 lowercase hex
    private static final List<Integer> TWELVE = List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

    @Autowired MockMvc mvc;
    @MockitoBean RecommendationService service;

    @Test
    void knownCustomer_returns200WithAlsList() throws Exception {
        when(service.recommend(ID, "repurchase")).thenReturn(
                new RecommendationResponse(ID, "repurchase", "als", "als-test", TWELVE));

        mvc.perform(get(URL, ID).param("mode", "repurchase"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.source").value("als"))
                .andExpect(jsonPath("$.mode").value("repurchase"))
                .andExpect(jsonPath("$.modelVersion").value("als-test"))
                .andExpect(jsonPath("$.items.length()").value(12));
    }

    @Test
    void unknownValidId_returns200WithFallback() throws Exception {
        // Shape check only. The real "lookup miss -> fallback" logic gets its own service test.
        when(service.recommend(ID, "discover")).thenReturn(
                new RecommendationResponse(ID, "discover", "fallback_popularity_7d", "als-test", TWELVE));

        mvc.perform(get(URL, ID).param("mode", "discover"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.source").value("fallback_popularity_7d"))
                .andExpect(jsonPath("$.mode").value("discover"));
    }

    @Test
    void malformedId_returns400AndNeverReachesService() throws Exception {
        mvc.perform(get(URL, "not-a-valid-id"))
                .andExpect(status().isBadRequest());
        verifyNoInteractions(service);
    }

    @Test
    void idWithLeadingSpace_returns400() throws Exception {
        mvc.perform(get(URL, " " + ID))
                .andExpect(status().isBadRequest());
        verifyNoInteractions(service);
    }

    @Test
    void uppercaseId_returns400() throws Exception {
        mvc.perform(get(URL, ID.toUpperCase()))
                .andExpect(status().isBadRequest());
        verifyNoInteractions(service);
    }

    @Test
    void invalidMode_returns400() throws Exception {
        mvc.perform(get(URL, ID).param("mode", "everything"))
                .andExpect(status().isBadRequest());
        verifyNoInteractions(service);
    }

    @Test
    void missingMode_defaultsToRepurchase() throws Exception {
        when(service.recommend(ID, "repurchase")).thenReturn(
                new RecommendationResponse(ID, "repurchase", "als", "als-test", TWELVE));

        mvc.perform(get(URL, ID))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.mode").value("repurchase"));
        verify(service).recommend(ID, "repurchase");
    }
@Test
    void fallbackUnavailable_returns503() throws Exception {
        when(service.recommend(ID, "repurchase"))
                .thenThrow(new FallbackUnavailableException("no fallback"));

        mvc.perform(get(URL, ID))
                .andExpect(status().isServiceUnavailable());
    }
}