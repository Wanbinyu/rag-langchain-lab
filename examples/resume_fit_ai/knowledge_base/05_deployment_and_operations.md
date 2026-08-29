# Resume Fit AI deployment and operations

Resume Fit AI runs on Node.js 22 or newer with Express. A small deployment can use PM2 to keep the process alive and Nginx or Caddy as a reverse proxy. A typical minimum host has one CPU core and 1 GB of memory; two cores and 2 GB provide more headroom for concurrent model requests and document parsing.

## Production startup checks

Production startup fails when no valid AI provider key is configured. Demo mode is allowed in development, but production accepts it only when `ALLOW_DEMO_MODE=true` is explicitly set. A custom provider endpoint must use HTTPS in production.

The service also refuses production startup when the operator name or contact information required by the legal pages is missing. API keys and the administrative statistics token belong only in a permission-restricted `.env` file and must never be embedded in frontend JavaScript.

## Health, limits, and cost controls

The health endpoint reports whether the configured AI connection is available. Operations should also verify process logs, privacy and terms pages, syntax checks, automated tests, and dependency audit results.

Rate limiting applies separately to analysis creation and file uploads. The default analysis rule allows ten task creations per IP in ten minutes. A configurable daily task limit and provider-side balance alerts reduce unexpected spend. The administrative endpoint exposes aggregate task counts, duration, success status, and token usage without returning resume or JD content.

## Scaling boundary

The current queue, daily counter, tasks, and short-lived results are process-local memory. This keeps the MVP simple and minimizes retained personal data, but it is not a multi-instance architecture. Horizontal scaling would require a shared queue, shared rate-limit state, durable or explicitly ephemeral result storage, distributed cancellation, and consistent task-token authorization.
