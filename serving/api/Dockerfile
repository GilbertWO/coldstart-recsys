# Container image for the recommendations API (Week 12).
# Two stages: build the jar with the Maven wrapper on a full JDK, then run it on a smaller JRE image.
# Build and run it through serving/docker-compose.yml:  docker compose --profile app up -d --build

FROM eclipse-temurin:21-jdk AS build
WORKDIR /app
# Dependencies first, so editing source code does not re-download them on every build.
COPY .mvn/ .mvn/
COPY mvnw pom.xml ./
RUN chmod +x mvnw && ./mvnw -B -q dependency:go-offline
COPY src/ src/
# The 33 tests run in GitHub Actions on every push; skipping them here keeps image builds fast.
RUN ./mvnw -B -q package -DskipTests && cp target/recs-api-*.jar app.jar

FROM eclipse-temurin:21-jre
# Run as an unprivileged user rather than root.
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY --from=build /app/app.jar app.jar
USER app
EXPOSE 8081
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
