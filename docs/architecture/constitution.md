---
id: constitution
title: ATLAS Constitution — Design Rules
status: stable
applies_to:
  - "docs/**"
  - "references/**"
last_validated: 2026-06-27
tags: [constitution, design-rules, invariants]
agent_hints:
  - "Read this before adding new content or reorganizing structure."
  - "Violating any rule below requires explicit justification in ADR."
---

# ATLAS Constitution

> Ironclad rules for maintaining ATLAS as a navigable, consistent, and AI-friendly knowledge base.

---

## Rule 1 — Document Completeness

**The rule.** Every topic document MUST include YAML frontmatter with `id`, `title`, `status`, and `last_validated`.

**Why.** Frontmatter enables programmatic filtering, status tracking, and automated validation. Without it, the knowledge base becomes unmaintainable at scale.

**How to apply.**
- Copy the template from `docs/adr/0000-template.md`
- Update `last_validated` on every meaningful edit
- Use status values: `draft`, `review`, `stable`, `deprecated`

**Where enforced.** `scripts/check_doc_consistency.py`

---

## Rule 2 — Cross-Reference Integrity

**The rule.** All internal links MUST use relative paths. Never use absolute paths.

**Why.** Relative paths survive repository moves and work across forks. Absolute paths break when the repo is cloned elsewhere.

**How to apply.**
- Use `[text](NN-topic.md#section)` format (template example)
- Always include file extension `.md`
- Run `scripts/check_doc_consistency.py` to validate

**Where enforced.** CI pre-commit hook and manual validation.

---

## Rule 3 — Layer Isolation

**The rule.** Each layer document should reference adjacent layers for context, but never assume implementation details from non-adjacent layers.

**Why.** Maintains modularity and prevents cascading updates when one layer changes. Foundation shouldn't know hardware specifics; Data-Ecosystem shouldn't assume model internals.

**How to apply.**
- Layer 01 references: 02, 05
- Layer 02 references: 01, 03
- Layer 03 references: 02, 04
- Layer 04 references: 03, 05
- Layer 05 references: 01, 04

**Where enforced.** Code review and architecture review.

---

## Rule 4 — Status Truthfulness

**The rule.** The `status:` field MUST accurately reflect document maturity. Never mark `stable` without validation.

**Why.** Status is used by AI agents and readers to gauge reliability. Misleading status erodes trust.

**How to apply.**
- `draft` - Initial content, structure may change
- `review` - Content complete, pending review
- `stable` - Reviewed, validated, ready for reference
- `deprecated` - Superseded, kept for history

**Where enforced.** `docs/_machine/status.yaml` tracks overall progress.

---

## Rule 5 — No Code Duplication

**The rule.** Never paste code blocks longer than 20 lines. Reference external repositories instead.

**Why.** Code rots. Pasted code becomes outdated and misleading. References stay current.

**How to apply.**
- Link to GitHub repos, papers, or official docs
- Include version/hash if specific version matters
- Explain the algorithm/concept, don't copy implementation

**Where enforced.** Code review for doc changes.

---

## Rule 6 — Terminology Consistency

**The rule.** Use consistent terminology throughout. Define terms in glossary and link to definitions.

**Why.** Inconsistent terminology confuses readers and breaks search.

**How to apply.**
- Check `docs/00-meta/glossary.md` before introducing new terms
- Use standard translations: "Ego-centric" not "第一视角主观"
- Add new terms to glossary before using them

**Where enforced.** `scripts/check_doc_consistency.py` flags undefined terms.

---

## Meta-Rule — Where new rules go

If you discover a new constraint that must hold across multiple subsystems, add it here AND record the incident / rationale that justifies it in an ADR.

For subsystem-local invariants, document them in the relevant `docs/subsystems/NN-*.md` file's `agent_hints` frontmatter.
