# PropelAI Ground Truth Annotation Guidelines

## Version 1.0

This document provides guidelines for annotating RFP requirements to create ground truth datasets for accuracy validation.

---

## 1. What is a Requirement?

A **requirement** is a statement that creates an obligation, expectation, or permission for the contractor or offeror.

### Include as Requirements:

1. **Mandatory obligations** - "The contractor SHALL provide..."
2. **Prohibitions** - "The contractor SHALL NOT disclose..."
3. **Expectations** - "The offeror SHOULD demonstrate..."
4. **Permissions** - "The contractor MAY use..."
5. **Evaluation criteria** - "Proposals WILL BE evaluated based on..."
6. **Deliverables** - "The contractor SHALL submit monthly reports..."
7. **Qualifications** - "Key personnel MUST have 5 years experience..."

### Exclude from Requirements:

1. **Headers and titles** - "SECTION C: STATEMENT OF WORK"
2. **Table of contents entries** - "3.1 Performance Requirements ... 15"
3. **Definitions** - "'COR' means Contracting Officer's Representative"
4. **Background information** - "The agency currently operates..."
5. **Government obligations** - "The Government shall provide access..."
6. **Boilerplate** - "This page intentionally left blank"
7. **Cross-references without obligation** - "See Attachment J-4"

---

## 2. Binding Levels

Assign ONE binding level to each requirement:

### Mandatory
- Keywords: SHALL, MUST, REQUIRED, WILL (with contractor as subject)
- Example: "The contractor SHALL deliver reports within 5 days"
- Note: Missing these = non-compliant proposal

### Highly Desirable
- Keywords: SHOULD, EXPECTED, RECOMMENDED
- Example: "Offerors SHOULD demonstrate relevant experience"
- Note: Not required but strongly influences evaluation

### Desirable
- Keywords: MAY, CAN, ENCOURAGED, PERMITTED, OPTIONAL
- Example: "The contractor MAY propose alternative approaches"
- Note: Optional, but can be discriminator

### Informational
- No binding keywords
- Context needed to understand other requirements
- Example: Evaluation factor descriptions without SHALL/SHOULD

---

## 3. Section Classification

Assign requirements to their source section:

| Section | Content |
|---------|---------|
| L | Instructions to Offerors (proposal format, submission requirements) |
| M | Evaluation Factors (how proposals will be scored) |
| C | Statement of Work / Performance Work Statement |
| B | Supplies or Services and Prices/Costs (CLINs) |
| F | Deliveries or Performance |
| G | Contract Administration Data |
| H | Special Contract Requirements |
| I | Contract Clauses |
| J | List of Attachments |
| K | Representations and Certifications |
| PWS | Performance Work Statement (may be Section C or attachment) |
| SOW | Statement of Work (may be Section C or attachment) |
| ATTACHMENT | Requirements from attachments not in above categories |

---

## 4. Requirement Categories

| Category | Description | Source Sections |
|----------|-------------|-----------------|
| L_COMPLIANCE | Proposal format, submission instructions | Section L |
| TECHNICAL | Performance requirements, deliverables | Section C, PWS, SOW |
| EVALUATION | Evaluation criteria, scoring factors | Section M |
| ADMINISTRATIVE | Contract admin, special requirements | B, F, G, H, I, K |
| ATTACHMENT | Requirements from attachments | Section J attachments |

---

## 5. Compound Requirements

Some sentences contain multiple requirements. Decompose them:

**Original**: "The contractor shall design, develop, test, and deploy the system."

**Decomposed**:
1. "The contractor shall design the system."
2. "The contractor shall develop the system."
3. "The contractor shall test the system."
4. "The contractor shall deploy the system."

**When to decompose**:
- Multiple distinct obligations in one sentence
- Different proposal sections would address each part
- Could be evaluated separately

**When NOT to decompose**:
- Items are inherently related (e.g., "design and implement")
- Would lose context if separated
- Enumerated list that's one logical requirement

---

## 6. Cross-References

Record when a requirement references other sections or documents:

**Example**: "In accordance with Section H, the contractor shall maintain CMMC Level 2 certification."

- Record "Section H" in `references_to` field
- Still extract as a standalone requirement

---

## 7. Source Location

For each requirement, record:

- **Page number**: 1-indexed page in the document
- **Section reference**: As it appears in RFP (e.g., "L.4.B.2", "C.3.1.a")
- **Source document**: Filename if multiple documents

---

## 8. Annotation Status

| Status | Meaning |
|--------|---------|
| draft | Initial annotation, needs review |
| reviewed | Reviewed by second annotator |
| approved | Final, ready for use as ground truth |
| disputed | Disagreement needs resolution |

---

## 9. Handling Ambiguity

When uncertain:

1. **Is it a requirement?**
   - If it creates ANY obligation → include it
   - If purely informational → exclude it
   - When in doubt → include and mark confidence < 1.0

2. **Which section?**
   - Use the section where text physically appears
   - If in attachment, use ATTACHMENT category

3. **Which binding level?**
   - Look for explicit keywords first
   - Context from surrounding text
   - Section M items without SHALL → still Mandatory to address

4. **Requirement boundaries?**
   - Err on the side of including complete context
   - Split only if truly independent obligations

---

## 10. Quality Checklist

Before marking a ground truth as complete:

- [ ] All SHALL/MUST statements extracted
- [ ] All SHOULD statements extracted
- [ ] All evaluation factors (Section M) extracted
- [ ] Attachments reviewed for requirements
- [ ] Compound requirements decomposed
- [ ] Section assignments verified
- [ ] Binding levels verified
- [ ] No duplicate requirements
- [ ] Second annotator review complete
- [ ] Inter-annotator disagreements resolved

---

## 11. Example Annotations

### Good Annotation

```json
{
  "gt_id": "GT-NIH-001",
  "text": "The Offeror shall provide a technical approach that describes the methods, tools, and processes to be used to accomplish the work.",
  "rfp_section": "L",
  "rfp_subsection": "L.4.B.2",
  "binding_level": "Mandatory",
  "binding_keyword": "shall",
  "category": "L_COMPLIANCE",
  "page_number": 47,
  "annotation_status": "approved"
}
```

### Annotation to Avoid

```json
{
  "gt_id": "GT-NIH-002",
  "text": "SECTION L - INSTRUCTIONS TO OFFERORS",
  "rfp_section": "L",
  "binding_level": "Mandatory",
  "category": "L_COMPLIANCE"
}
```
**Problem**: This is a header, not a requirement.

---

## 12. Inter-Annotator Agreement

Target agreement levels:

| Metric | Target |
|--------|--------|
| Requirement detection (Kappa) | > 0.85 |
| Section attribution | > 0.90 |
| Binding level | > 0.95 |

When annotators disagree:
1. Document both interpretations
2. Third annotator (senior) makes final decision
3. Record rationale in `resolution_notes`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-14 | Initial guidelines |
