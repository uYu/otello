FROM caddy:2 AS web-base
COPY deploy/Caddyfile /etc/caddy/Caddyfile
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD wget -q -O /dev/null http://127.0.0.1/opening-book/0dd454d84aa6-v1/manifest.json || exit 1

# Deploy a locally verified build without installing build tools on the server.
FROM web-base AS prebuilt
COPY dist/ /srv/

FROM node:22-alpine AS build
WORKDIR /app
RUN apk add --no-cache python3 g++
COPY package*.json ./
RUN npm ci
COPY index.html tsconfig.json vite.config.ts ./
COPY src/ ./src/
COPY public/ ./public/
COPY models/default-model.json models/v1.36-default.json models/pattern32-outcome-v1.25.json models/pattern32-outcome-v1.25.otpat models/training-data-manifest.json ./models/
COPY data/README.md ./data/README.md
COPY data/manifest.json ./data/manifest.json
COPY data/opening-book/egaroucid-2a65c87860be.tar.gz ./data/opening-book/egaroucid-2a65c87860be.tar.gz
COPY cpp/othello.hpp ./cpp/othello.hpp
COPY tools/export-web-book.cpp ./tools/export-web-book.cpp
COPY scripts/check-web-release.mjs scripts/prepare-web-release.mjs scripts/prepare-web-book.py scripts/data-assets.py ./scripts/
RUN npm run build

FROM web-base AS production
COPY --from=build /app/dist/ /srv/
