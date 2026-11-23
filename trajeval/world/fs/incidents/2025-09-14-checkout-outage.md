# Incident report: checkout outage, 14 September 2025

**Severity:** SEV-1
**Duration:** 19:40 to 21:05 IST (85 minutes)
**Incident owner:** Abhay Thakur (Backend)

## Impact
Customers could add items to the cart but every payment confirmation failed. About 140
checkout attempts failed during the window. Roughly Rs 6.2 lakh of orders were delayed; most
customers retried later, and 23 orders were never completed.

## Timeline
- 19:40 Payment webhook calls start failing with TLS handshake errors.
- 19:58 On-call (Kunal Jain) paged by the checkout error-rate alert.
- 20:35 Root cause identified.
- 20:52 New certificate deployed.
- 21:05 Error rate back to baseline.

## Root cause
The TLS certificate on the payment webhook endpoint (pay-hooks.kestrelcycles.example)
expired. Its auto-renewal cron job had been disabled during the server migration on
2 September and was never re-enabled.

## Fix and follow-ups
1. Renewed the certificate and re-enabled the renewal cron job (done).
2. Added an alert that fires 14 days before any certificate expires (done).
3. Add certificate checks to the migration checklist (owner: Ritu Bhatia, due 30 Sep).
