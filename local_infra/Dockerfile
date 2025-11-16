# syntax=docker/dockerfile:1

##############################
# Build Habitat's web assets #
##############################
FROM node:20-alpine AS ui-build
WORKDIR /build
# Copy habitat directory structure
COPY third_party/absurd/habitat/ .
# Install and build UI (outputs to internal/web/dist per vite.config.ts)
RUN cd ui && npm ci && npm run build

############################
# Build the Habitat binary #
############################
FROM golang:1.24.2-alpine AS build
WORKDIR /src
# copy the vendored absurd repo
COPY third_party/absurd/ /src/
# drop in the built UI assets so go:embed finds dist files
COPY --from=ui-build /build/internal/web/dist /src/habitat/internal/web/dist
# build Habitat (Go web UI)
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd habitat && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/habitat ./cmd/habitat

#########################################
# Final image: Postgres + Supervisor    #
#########################################
FROM postgres:16-alpine
RUN apk add --no-cache supervisor

# Seed Absurd’s schema on first init of the data dir
COPY third_party/absurd/sql/absurd.sql /docker-entrypoint-initdb.d/01_absurd.sql

# Drop in the Habitat binary
COPY --from=build /out/habitat /usr/local/bin/habitat

# Supervisor config
COPY local_infra/supervisord.conf /etc/supervisord.conf

# Default env (can be overridden by compose)
ENV POSTGRES_USER=absurd \
    POSTGRES_PASSWORD=absurd \
    POSTGRES_DB=absurd \
    PGDATABASE=postgresql://absurd:absurd@127.0.0.1:5432/absurd?sslmode=disable \
    DATABASE_URL=postgresql://absurd:absurd@127.0.0.1:5432/absurd?sslmode=disable \
    PORT=8080

EXPOSE 5432 8080

# Run both Postgres and Habitat
CMD ["supervisord","-n","-c","/etc/supervisord.conf"]
