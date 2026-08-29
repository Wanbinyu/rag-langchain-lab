# Resume Fit AI asynchronous task lifecycle

AI analysis runs through an in-memory task queue instead of holding one HTTP request open until model completion. The create endpoint validates input, rate limits the caller, applies an idempotency key, and returns a task identifier plus a separate random access token. Knowing a task identifier alone is not sufficient to read its report.

## Queue controls

The default configuration allows two AI requests to run concurrently and up to twenty tasks to wait in the queue. A configurable daily task ceiling limits accidental spending. Because these counters and tasks live in process memory, a server restart resets them and a multi-instance deployment would require shared state such as Redis and a durable job system.

Clients poll task progress and can recover polling after a page refresh. A queued or running task can be cancelled. Capacity errors are returned before an expensive model call is started.

## Retry policy

Automatic retry is limited to temporary failures: timeouts, HTTP 429 rate limits, and upstream 5xx responses. Validation errors and permanent provider failures are not retried. Retries are bounded, and the idempotency key prevents a browser or network retry from creating duplicate AI jobs.

## Data lifetime

The raw resume and JD are cleared from task memory immediately after completion, failure, or cancellation. A generated result remains available for at most fifteen minutes by default and is then removed. Resume editing history is kept in the current browser's local storage rather than a server database, so the user must export important work before clearing browser data.

Anonymous operational statistics contain status, duration, and token usage, but not resume text, JD text, or generated reports.
