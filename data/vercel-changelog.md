---
url: https://vercel.com/changelog
fetched_at: 2026-11-09
title: Changelog — Vercel
synthetic: true
---
# Changelog

New features, improvements, and fixes to the Vercel platform.

## November 4, 2026 — Vercel Agent Runtime is now generally available
Vercel Agent Runtime lets you deploy long-running AI agents next to your frontend with one command. Agents get durable execution: they can pause for hours while waiting for a tool call or human approval and resume where they left off, with no extra infrastructure. Each step is traced in the Observability tab, and costs are billed per active CPU-millisecond, so idle waiting is free. Agent Runtime is available on all plans; Pro and Enterprise teams get higher concurrency limits.

## October 28, 2026 — Improved build cache for monorepos
Turborepo-backed builds now share a remote cache across all projects in the same team. Median build times for large monorepos dropped 34% in our beta.

## October 21, 2026 — Firewall: custom rate-limit rules on Hobby
Hobby teams can now create up to three custom rate-limit rules in Vercel Firewall. Previously, custom rules were available only on Pro and Enterprise.

## October 9, 2026 — AI Gateway adds bring-your-own-key routing
AI Gateway can now route requests through your own provider API keys while keeping unified logging, fallbacks, and spend limits. Requests made with your own keys are not marked up.

## September 30, 2026 — Vercel Queues is now generally available
Vercel Queues gives you durable, at-least-once message delivery between functions, with automatic retries and dead-letter handling. Queues was in public beta since May 2026.

## September 17, 2026 — Node.js 24 is the new default runtime
New projects now default to Node.js 24. Existing projects keep their configured version.
