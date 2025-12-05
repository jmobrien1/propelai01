/**
 * PropelAI: Annotated Outline Exporter v1.0
 * 
 * Generates professional Word documents for proposal development following
 * industry best practices (Shipley Associates, Lohfeld Consulting Group).
 * 
 * The Annotated Outline is the "single most important tool" for government
 * proposals - it serves as the architectural blueprint for a winning response.
 * 
 * Features:
 * - Structure mirrors Section L exactly (evaluator navigation)
 * - Color-coded requirements (Red=L instructions, Blue=M evaluation, Purple=C/PWS)
 * - Page allocations per section
 * - Win theme & discriminator placeholders
 * - Proof point guidance
 * - Graphics placeholders with action caption templates
 * - Boilerplate direction
 * - Formatting requirements from RFP
 */

const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, PageOrientation, LevelFormat,
        HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber,
        PageBreak, TabStopType, TabStopPosition } = require('docx');
const fs = require('fs');

// Color scheme per best practices
const COLORS = {
    SECTION_L: "C00000",      // Red - Instructions (how to submit)
    SECTION_M: "0070C0",      // Blue - Evaluation (how scored)
    SECTION_C: "7030A0",      // Purple - PWS/SOW (what to do)
    WIN_THEME: "00B050",      // Green - Win themes/discriminators
    PROOF_POINT: "ED7D31",    // Orange - Proof points needed
    GRAPHIC: "5B9BD5",        // Light blue - Graphics placeholders
    BOILERPLATE: "808080",    // Gray - Boilerplate guidance
    BLACK: "000000",
    DARK_GRAY: "404040"
};

// Section annotation box styling
const ANNOTATION_SHADING = {
    L: { fill: "FCE4D6", type: ShadingType.CLEAR },  // Light red/peach
    M: { fill: "D6E3F8", type: ShadingType.CLEAR },  // Light blue
    C: { fill: "E4D5F0", type: ShadingType.CLEAR },  // Light purple
    STRATEGY: { fill: "E2F0D9", type: ShadingType.CLEAR },  // Light green
    PROOF: { fill: "FFF2CC", type: ShadingType.CLEAR },     // Light yellow/orange
    GRAPHIC: { fill: "DEEBF7", type: ShadingType.CLEAR }    // Light blue-gray
};

/**
 * Main function to generate annotated outline document
 */
function generateAnnotatedOutline(data) {
    const {
        rfpTitle = "RFP Title",
        solicitationNumber = "TBD",
        dueDate = "TBD",
        submissionMethod = "Not Specified",
        totalPages = null,
        formatRequirements = {},
        volumes = [],
        evaluationFactors = [],
        requirements = [],         // From CTM extraction
        documentStructure = null,  // From document_structure parser
        winThemes = [],            // Optional: pre-defined win themes
        companyName = "[Company Name]",
        // OASIS+ / P0 Constraint data
        p0Constraints = [],        // P0 formatting constraints from extraction
        mandatoryArtifacts = [],   // Mandatory forms for Volume 3
        adjectivalRatings = {}     // Rating scale definitions
    } = data;

    // Merge p0 constraints from formatRequirements if present
    const allP0Constraints = [
        ...(p0Constraints || []),
        ...(formatRequirements.p0_constraints || [])
    ];

    // Build the document
    const doc = new Document({
        styles: {
            default: {
                document: {
                    run: { font: formatRequirements.font || "Arial", size: 22 }  // 11pt default
                }
            },
            paragraphStyles: [
                {
                    id: "Title",
                    name: "Title",
                    basedOn: "Normal",
                    run: { size: 48, bold: true, color: COLORS.BLACK, font: "Arial" },
                    paragraph: { spacing: { before: 0, after: 240 }, alignment: AlignmentType.CENTER }
                },
                {
                    id: "Heading1",
                    name: "Heading 1",
                    basedOn: "Normal",
                    next: "Normal",
                    quickFormat: true,
                    run: { size: 32, bold: true, color: COLORS.BLACK, font: "Arial" },
                    paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 }
                },
                {
                    id: "Heading2",
                    name: "Heading 2",
                    basedOn: "Normal",
                    next: "Normal",
                    quickFormat: true,
                    run: { size: 26, bold: true, color: COLORS.DARK_GRAY, font: "Arial" },
                    paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
                },
                {
                    id: "Heading3",
                    name: "Heading 3",
                    basedOn: "Normal",
                    next: "Normal",
                    quickFormat: true,
                    run: { size: 24, bold: true, color: COLORS.DARK_GRAY, font: "Arial" },
                    paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 }
                },
                {
                    id: "AnnotationLabel",
                    name: "Annotation Label",
                    basedOn: "Normal",
                    run: { size: 18, bold: true, smallCaps: true, font: "Arial" },
                    paragraph: { spacing: { before: 60, after: 40 } }
                },
                {
                    id: "RequirementText",
                    name: "Requirement Text",
                    basedOn: "Normal",
                    run: { size: 20, italics: true, font: "Arial" },
                    paragraph: { spacing: { before: 40, after: 40 }, indent: { left: 360 } }
                },
                {
                    id: "WriterGuidance",
                    name: "Writer Guidance",
                    basedOn: "Normal",
                    run: { size: 20, font: "Arial", color: COLORS.DARK_GRAY },
                    paragraph: { spacing: { before: 80, after: 80 }, indent: { left: 360 } }
                }
            ]
        },
        numbering: {
            config: [
                {
                    reference: "section-bullets",
                    levels: [{
                        level: 0,
                        format: LevelFormat.BULLET,
                        text: "•",
                        alignment: AlignmentType.LEFT,
                        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
                    }]
                }
            ]
        },
        sections: [{
            properties: {
                page: {
                    margin: {
                        top: formatRequirements.margins ? 1440 : 1440,
                        right: 1440,
                        bottom: 1440,
                        left: 1440
                    },
                    size: { orientation: PageOrientation.PORTRAIT }
                }
            },
            headers: {
                default: new Header({
                    children: [
                        new Paragraph({
                            alignment: AlignmentType.RIGHT,
                            children: [
                                new TextRun({ text: `${rfpTitle} - ANNOTATED OUTLINE`, size: 18, color: COLORS.DARK_GRAY }),
                            ]
                        })
                    ]
                })
            },
            footers: {
                default: new Footer({
                    children: [
                        new Paragraph({
                            alignment: AlignmentType.CENTER,
                            children: [
                                new TextRun({ text: `${companyName} PROPRIETARY - `, size: 16, color: COLORS.DARK_GRAY }),
                                new TextRun({ text: "Page ", size: 16 }),
                                new TextRun({ children: [PageNumber.CURRENT], size: 16 }),
                                new TextRun({ text: " of ", size: 16 }),
                                new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16 })
                            ]
                        })
                    ]
                })
            },
            children: buildDocumentContent({
                rfpTitle,
                solicitationNumber,
                dueDate,
                submissionMethod,
                totalPages,
                formatRequirements,
                volumes,
                evaluationFactors,
                requirements,
                documentStructure,
                winThemes,
                companyName,
                p0Constraints: allP0Constraints,
                mandatoryArtifacts,
                adjectivalRatings
            })
        }]
    });

    return doc;
}

/**
 * Build the main document content
 */
function buildDocumentContent(data) {
    const children = [];

    // === COVER PAGE ===
    children.push(...buildCoverPage(data));
    children.push(new Paragraph({ children: [new PageBreak()] }));

    // === FORMATTING REQUIREMENTS ===
    children.push(...buildFormatRequirementsSection(data));

    // === P0 CONSTRAINTS (CRITICAL) ===
    if (data.p0Constraints && data.p0Constraints.length > 0) {
        children.push(...buildP0ConstraintsSection(data));
    }
    children.push(new Paragraph({ children: [new PageBreak()] }));

    // === COLOR CODE LEGEND ===
    children.push(...buildColorLegend());

    // === EVALUATION FACTORS SUMMARY ===
    children.push(...buildEvalFactorsSummary(data));

    // === ADJECTIVAL RATING DEFINITIONS (for OASIS+) ===
    if (data.adjectivalRatings && Object.keys(data.adjectivalRatings).length > 0) {
        children.push(...buildAdjectivalRatingsSection(data));
    }
    children.push(new Paragraph({ children: [new PageBreak()] }));

    // === VOLUME OUTLINES ===
    children.push(...buildVolumeOutlines(data));

    // === VOLUME 3: CONTRACT DOCUMENTATION (if mandatory artifacts exist) ===
    if (data.mandatoryArtifacts && data.mandatoryArtifacts.length > 0) {
        children.push(...buildVolume3ContractDocs(data));
    }

    return children;
}

/**
 * Build cover page
 */
function buildCoverPage(data) {
    return [
        new Paragraph({ spacing: { before: 2000 } }),
        new Paragraph({
            heading: HeadingLevel.TITLE,
            children: [new TextRun({ text: "ANNOTATED PROPOSAL OUTLINE" })]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 400, after: 400 },
            children: [
                new TextRun({ text: data.rfpTitle, size: 36, bold: true, color: COLORS.BLACK })
            ]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
            children: [
                new TextRun({ text: `Solicitation Number: ${data.solicitationNumber}`, size: 24 })
            ]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 600 },
            children: [
                new TextRun({ text: `Proposal Due: ${data.dueDate}`, size: 24, bold: true, color: COLORS.SECTION_L })
            ]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 800 },
            children: [
                new TextRun({ text: `Prepared by: ${data.companyName}`, size: 22 })
            ]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 1200, after: 400 },
            border: { top: { style: BorderStyle.SINGLE, size: 6, color: COLORS.BLACK } },
            children: [
                new TextRun({
                    text: "This annotated outline serves as the master template for all proposal content development. ",
                    size: 20, italics: true
                }),
                new TextRun({
                    text: "All writers must address every annotated requirement and follow the page allocations specified.",
                    size: 20, italics: true
                })
            ]
        })
    ];
}

/**
 * Build formatting requirements section
 */
function buildFormatRequirementsSection(data) {
    const fmt = data.formatRequirements || {};
    const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
    const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

    return [
        new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun({ text: "SUBMISSION & FORMAT REQUIREMENTS" })]
        }),
        new Paragraph({
            spacing: { after: 200 },
            children: [
                new TextRun({
                    text: "CRITICAL: Non-compliance with format requirements may result in proposal disqualification.",
                    bold: true, color: COLORS.SECTION_L
                })
            ]
        }),
        new Table({
            columnWidths: [3000, 6360],
            rows: [
                createTableRow("Due Date", data.dueDate, cellBorders, true),
                createTableRow("Submission Method", data.submissionMethod, cellBorders),
                createTableRow("Total Page Limit", data.totalPages ? `${data.totalPages} pages` : "See individual volumes", cellBorders),
                createTableRow("Font", fmt.font || "Verify in RFP", cellBorders),
                createTableRow("Font Size", fmt.font_size ? `${fmt.font_size} pt` : "Verify in RFP", cellBorders),
                createTableRow("Margins", fmt.margins || "Verify in RFP", cellBorders),
                createTableRow("Line Spacing", fmt.line_spacing || "Verify in RFP", cellBorders),
                createTableRow("Page Size", fmt.page_size || "8.5\" x 11\" (Letter)", cellBorders)
            ]
        }),
        new Paragraph({ spacing: { after: 200 } })
    ];
}

/**
 * Build P0 Constraints Section - CRITICAL disqualification risks
 */
function buildP0ConstraintsSection(data) {
    const constraints = data.p0Constraints || [];
    const tableBorder = { style: BorderStyle.SINGLE, size: 2, color: COLORS.SECTION_L };
    const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

    const children = [
        new Paragraph({
            heading: HeadingLevel.HEADING_2,
            children: [
                new TextRun({ text: "⚠️ P0 CONSTRAINTS - DISQUALIFICATION RISKS", color: COLORS.SECTION_L })
            ]
        }),
        new Paragraph({
            spacing: { after: 120 },
            shading: { fill: "FCE4D6", type: ShadingType.CLEAR },
            border: { left: { style: BorderStyle.SINGLE, size: 12, color: COLORS.SECTION_L } },
            indent: { left: 180 },
            children: [
                new TextRun({
                    text: "CRITICAL: The following constraints are Pass/Fail. Violation of ANY P0 constraint may result in proposal rejection without further evaluation.",
                    bold: true, size: 22, color: COLORS.SECTION_L
                })
            ]
        })
    ];

    // Build constraint table
    const rows = [
        new TableRow({
            tableHeader: true,
            children: [
                new TableCell({
                    borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                    shading: { fill: COLORS.SECTION_L, type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Type", bold: true, color: "FFFFFF", size: 18 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 3500, type: WidthType.DXA },
                    shading: { fill: COLORS.SECTION_L, type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Constraint", bold: true, color: "FFFFFF", size: 18 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 1800, type: WidthType.DXA },
                    shading: { fill: COLORS.SECTION_L, type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Applies To", bold: true, color: "FFFFFF", size: 18 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 2560, type: WidthType.DXA },
                    shading: { fill: COLORS.SECTION_L, type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Consequence", bold: true, color: "FFFFFF", size: 18 })] })]
                })
            ]
        })
    ];

    constraints.forEach(constraint => {
        const typeLabel = constraint.type || constraint.constraint_type || "UNKNOWN";
        const description = constraint.description || "";
        const appliesTo = constraint.applies_to || "All";
        const consequence = constraint.consequence || "May cause disqualification";

        rows.push(new TableRow({
            children: [
                new TableCell({
                    borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                    children: [new Paragraph({ children: [new TextRun({ text: typeLabel, bold: true, size: 18 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 3500, type: WidthType.DXA },
                    children: [new Paragraph({ children: [new TextRun({ text: description, size: 18 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 1800, type: WidthType.DXA },
                    children: [new Paragraph({ children: [new TextRun({ text: appliesTo, size: 18 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 2560, type: WidthType.DXA },
                    shading: { fill: "FCE4D6", type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: consequence, size: 18, bold: true, color: COLORS.SECTION_L })] })]
                })
            ]
        }));
    });

    children.push(new Table({ columnWidths: [1500, 3500, 1800, 2560], rows }));
    children.push(new Paragraph({ spacing: { after: 200 } }));

    return children;
}

/**
 * Build Adjectival Ratings Section for OASIS+ task orders
 */
function buildAdjectivalRatingsSection(data) {
    const ratings = data.adjectivalRatings || {};
    const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: COLORS.SECTION_M };
    const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

    const children = [
        new Paragraph({
            heading: HeadingLevel.HEADING_2,
            children: [new TextRun({ text: "Adjectival Rating Definitions" })]
        }),
        new Paragraph({
            spacing: { after: 120 },
            children: [new TextRun({
                text: "Understanding the rating scale is essential for targeting Exceptional ratings on key factors.",
                size: 20, color: COLORS.SECTION_M
            })]
        })
    ];

    const ratingOrder = ['Exceptional', 'Very Good', 'Good', 'Unacceptable'];
    const ratingColors = {
        'Exceptional': '00B050',  // Green
        'Very Good': '92D050',    // Light green
        'Good': 'FFC000',         // Yellow
        'Unacceptable': 'C00000'  // Red
    };

    const rows = [
        new TableRow({
            tableHeader: true,
            children: [
                new TableCell({
                    borders: cellBorders, width: { size: 2000, type: WidthType.DXA },
                    shading: { fill: COLORS.SECTION_M, type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Rating", bold: true, color: "FFFFFF", size: 20 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 7360, type: WidthType.DXA },
                    shading: { fill: COLORS.SECTION_M, type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Definition", bold: true, color: "FFFFFF", size: 20 })] })]
                })
            ]
        })
    ];

    ratingOrder.forEach(rating => {
        const definition = ratings[rating] || ratings[rating.toLowerCase()] || `See RFP for ${rating} definition`;
        const color = ratingColors[rating] || COLORS.BLACK;

        rows.push(new TableRow({
            children: [
                new TableCell({
                    borders: cellBorders, width: { size: 2000, type: WidthType.DXA },
                    children: [new Paragraph({ children: [new TextRun({ text: rating, bold: true, size: 20, color: color })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 7360, type: WidthType.DXA },
                    children: [new Paragraph({ children: [new TextRun({ text: definition, size: 18 })] })]
                })
            ]
        }));
    });

    children.push(new Table({ columnWidths: [2000, 7360], rows }));
    children.push(new Paragraph({ spacing: { after: 200 } }));

    return children;
}

/**
 * Build Volume 3: Contract Documentation section with mandatory artifacts
 */
function buildVolume3ContractDocs(data) {
    const artifacts = data.mandatoryArtifacts || [];
    const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "666666" };
    const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

    const children = [
        new Paragraph({
            heading: HeadingLevel.HEADING_1,
            pageBreakBefore: true,
            children: [new TextRun({ text: "VOLUME 3: CONTRACT DOCUMENTATION" })]
        }),
        new Paragraph({
            spacing: { after: 120 },
            shading: { fill: "FCE4D6", type: ShadingType.CLEAR },
            border: { left: { style: BorderStyle.SINGLE, size: 12, color: COLORS.SECTION_L } },
            indent: { left: 180 },
            children: [
                new TextRun({
                    text: "PASS/FAIL: All items in this volume are mandatory. Missing or incomplete documents may result in proposal rejection.",
                    bold: true, size: 20, color: COLORS.SECTION_L
                })
            ]
        }),
        new Paragraph({
            heading: HeadingLevel.HEADING_2,
            children: [new TextRun({ text: "Mandatory Forms and Certifications" })]
        })
    ];

    // Build artifact checklist table
    const rows = [
        new TableRow({
            tableHeader: true,
            children: [
                new TableCell({
                    borders: cellBorders, width: { size: 500, type: WidthType.DXA },
                    shading: { fill: "D5D5D5", type: ShadingType.CLEAR },
                    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "✓", bold: true, size: 20 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 2200, type: WidthType.DXA },
                    shading: { fill: "D5D5D5", type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Document", bold: true, size: 20 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 4000, type: WidthType.DXA },
                    shading: { fill: "D5D5D5", type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Description / Instructions", bold: true, size: 20 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 1600, type: WidthType.DXA },
                    shading: { fill: "D5D5D5", type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Reference", bold: true, size: 20 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 1060, type: WidthType.DXA },
                    shading: { fill: "D5D5D5", type: ShadingType.CLEAR },
                    children: [new Paragraph({ children: [new TextRun({ text: "Status", bold: true, size: 20 })] })]
                })
            ]
        })
    ];

    artifacts.forEach(artifact => {
        const name = artifact.name || artifact.id || "Document";
        const description = artifact.description || "";
        const farRef = artifact.far_reference || artifact.form_number || "";
        const isPassFail = artifact.is_pass_fail !== false;

        rows.push(new TableRow({
            children: [
                new TableCell({
                    borders: cellBorders, width: { size: 500, type: WidthType.DXA },
                    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "☐", size: 24 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 2200, type: WidthType.DXA },
                    children: [new Paragraph({ children: [
                        new TextRun({ text: name, bold: true, size: 18 }),
                        isPassFail ? new TextRun({ text: " [P/F]", size: 16, color: COLORS.SECTION_L }) : new TextRun({ text: "" })
                    ] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 4000, type: WidthType.DXA },
                    children: [new Paragraph({ children: [new TextRun({ text: description, size: 18 })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 1600, type: WidthType.DXA },
                    children: [new Paragraph({ children: [new TextRun({ text: farRef, size: 18, color: COLORS.SECTION_M })] })]
                }),
                new TableCell({
                    borders: cellBorders, width: { size: 1060, type: WidthType.DXA },
                    children: [new Paragraph({ children: [new TextRun({ text: "Pending", size: 16, italics: true, color: COLORS.DARK_GRAY })] })]
                })
            ]
        }));
    });

    children.push(new Table({ columnWidths: [500, 2200, 4000, 1600, 1060], rows }));

    // Add guidance for Volume 3 completion
    children.push(new Paragraph({ spacing: { before: 200 } }));
    children.push(createAnnotationBlock(
        "VOLUME 3 COMPLETION CHECKLIST",
        [
            "☐ SF1449 - Completed with all required fields (Blocks 12-17)",
            "☐ Online Representations - Verify SAM.gov registration is current",
            "☐ FAR 52.204-7 - Submit evidence of registration",
            "☐ Security Forms - DD-254 if applicable (verify classification requirements)",
            "☐ Certifications - All agency-specific certifications completed",
            "☐ Subcontracting Plan - Required if over simplified acquisition threshold"
        ],
        COLORS.SECTION_L,
        ANNOTATION_SHADING.L,
        false
    ));

    return children;
}

function createTableRow(label, value, borders, highlight = false) {
    const labelShading = highlight ? { fill: "FFC000", type: ShadingType.CLEAR } : undefined;
    return new TableRow({
        children: [
            new TableCell({
                borders,
                width: { size: 3000, type: WidthType.DXA },
                shading: labelShading,
                children: [new Paragraph({
                    children: [new TextRun({ text: label, bold: true, size: 22 })]
                })]
            }),
            new TableCell({
                borders,
                width: { size: 6360, type: WidthType.DXA },
                shading: labelShading,
                children: [new Paragraph({
                    children: [new TextRun({
                        text: value || "TBD",
                        size: 22,
                        color: value ? COLORS.BLACK : COLORS.SECTION_L
                    })]
                })]
            })
        ]
    });
}

/**
 * Build color code legend
 */
function buildColorLegend() {
    const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
    const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

    return [
        new Paragraph({
            heading: HeadingLevel.HEADING_2,
            children: [new TextRun({ text: "Annotation Color Code Legend" })]
        }),
        new Paragraph({
            spacing: { after: 120 },
            children: [new TextRun({
                text: "Requirements are color-coded by source to help writers understand what the customer is asking for (Section C), how to submit it (Section L), and how it will be scored (Section M).",
                size: 20
            })]
        }),
        new Table({
            columnWidths: [1800, 3000, 4560],
            rows: [
                new TableRow({
                    children: [
                        new TableCell({
                            borders: cellBorders, width: { size: 1800, type: WidthType.DXA },
                            shading: { fill: "D5D5D5", type: ShadingType.CLEAR },
                            children: [new Paragraph({ children: [new TextRun({ text: "Color", bold: true, size: 20 })] })]
                        }),
                        new TableCell({
                            borders: cellBorders, width: { size: 3000, type: WidthType.DXA },
                            shading: { fill: "D5D5D5", type: ShadingType.CLEAR },
                            children: [new Paragraph({ children: [new TextRun({ text: "Source", bold: true, size: 20 })] })]
                        }),
                        new TableCell({
                            borders: cellBorders, width: { size: 4560, type: WidthType.DXA },
                            shading: { fill: "D5D5D5", type: ShadingType.CLEAR },
                            children: [new Paragraph({ children: [new TextRun({ text: "Purpose", bold: true, size: 20 })] })]
                        })
                    ]
                }),
                createLegendRow("RED", "Section L (Instructions)", "How to submit - format, content, page limits", COLORS.SECTION_L, cellBorders),
                createLegendRow("BLUE", "Section M (Evaluation)", "How proposal will be scored - criteria, weights", COLORS.SECTION_M, cellBorders),
                createLegendRow("PURPLE", "Section C/PWS/SOW", "What to do - technical requirements, scope", COLORS.SECTION_C, cellBorders),
                createLegendRow("GREEN", "Win Themes", "Discriminators and key messages to emphasize", COLORS.WIN_THEME, cellBorders),
                createLegendRow("ORANGE", "Proof Points", "Evidence needed - past performance, metrics, certs", COLORS.PROOF_POINT, cellBorders),
                createLegendRow("BLUE-GRAY", "Graphics", "Planned visuals with action captions", COLORS.GRAPHIC, cellBorders)
            ]
        }),
        new Paragraph({ spacing: { after: 300 } })
    ];
}

function createLegendRow(colorName, source, purpose, hexColor, borders) {
    return new TableRow({
        children: [
            new TableCell({
                borders, width: { size: 1800, type: WidthType.DXA },
                children: [new Paragraph({
                    children: [new TextRun({ text: colorName, bold: true, color: hexColor, size: 20 })]
                })]
            }),
            new TableCell({
                borders, width: { size: 3000, type: WidthType.DXA },
                children: [new Paragraph({
                    children: [new TextRun({ text: source, size: 20 })]
                })]
            }),
            new TableCell({
                borders, width: { size: 4560, type: WidthType.DXA },
                children: [new Paragraph({
                    children: [new TextRun({ text: purpose, size: 20 })]
                })]
            })
        ]
    });
}

/**
 * Build evaluation factors summary
 */
function buildEvalFactorsSummary(data) {
    const factors = data.evaluationFactors || [];
    const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
    const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

    const children = [
        new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun({ text: "EVALUATION FACTORS SUMMARY" })]
        }),
        new Paragraph({
            spacing: { after: 200 },
            children: [new TextRun({
                text: "Understanding how the proposal will be scored is critical. Structure content to directly address each evaluation factor.",
                size: 20, color: COLORS.SECTION_M
            })]
        })
    ];

    if (factors.length > 0) {
        const rows = [
            new TableRow({
                tableHeader: true,
                children: [
                    new TableCell({
                        borders: cellBorders, width: { size: 2800, type: WidthType.DXA },
                        shading: { fill: COLORS.SECTION_M, type: ShadingType.CLEAR },
                        children: [new Paragraph({ children: [new TextRun({ text: "Factor", bold: true, color: "FFFFFF", size: 20 })] })]
                    }),
                    new TableCell({
                        borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                        shading: { fill: COLORS.SECTION_M, type: ShadingType.CLEAR },
                        children: [new Paragraph({ children: [new TextRun({ text: "Weight", bold: true, color: "FFFFFF", size: 20 })] })]
                    }),
                    new TableCell({
                        borders: cellBorders, width: { size: 5060, type: WidthType.DXA },
                        shading: { fill: COLORS.SECTION_M, type: ShadingType.CLEAR },
                        children: [new Paragraph({ children: [new TextRun({ text: "Key Criteria", bold: true, color: "FFFFFF", size: 20 })] })]
                    })
                ]
            })
        ];

        factors.forEach(factor => {
            rows.push(new TableRow({
                children: [
                    new TableCell({
                        borders: cellBorders, width: { size: 2800, type: WidthType.DXA },
                        children: [new Paragraph({ children: [new TextRun({ text: factor.name || factor.id, bold: true, size: 20 })] })]
                    }),
                    new TableCell({
                        borders: cellBorders, width: { size: 1500, type: WidthType.DXA },
                        children: [new Paragraph({ children: [new TextRun({ text: factor.weight || factor.importance || "TBD", size: 20 })] })]
                    }),
                    new TableCell({
                        borders: cellBorders, width: { size: 5060, type: WidthType.DXA },
                        children: [new Paragraph({
                            children: [new TextRun({
                                text: (factor.criteria || []).join("; ") || "See Section M for details",
                                size: 20
                            })]
                        })]
                    })
                ]
            }));
        });

        children.push(new Table({ columnWidths: [2800, 1500, 5060], rows }));
    } else {
        children.push(new Paragraph({
            children: [new TextRun({
                text: "⚠ No evaluation factors extracted. Review Section M of the RFP manually.",
                color: COLORS.SECTION_L, bold: true, size: 22
            })]
        }));
    }

    return children;
}

/**
 * Build volume outlines with full annotations
 */
function buildVolumeOutlines(data) {
    const volumes = data.volumes || [];
    const requirements = data.requirements || [];
    const children = [];

    volumes.forEach((volume, volIndex) => {
        // Volume header
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_1,
            pageBreakBefore: volIndex > 0,
            children: [
                new TextRun({ text: `VOLUME ${volIndex + 1}: ${volume.name.toUpperCase()}` })
            ]
        }));

        // Volume metadata
        children.push(new Paragraph({
            spacing: { after: 120 },
            children: [
                new TextRun({ text: "Page Limit: ", bold: true, size: 22 }),
                new TextRun({
                    text: volume.page_limit ? `${volume.page_limit} pages` : "No limit specified",
                    size: 22,
                    color: volume.page_limit ? COLORS.BLACK : COLORS.SECTION_L
                })
            ]
        }));

        if (volume.eval_factors && volume.eval_factors.length > 0) {
            children.push(new Paragraph({
                spacing: { after: 200 },
                children: [
                    new TextRun({ text: "Evaluation Factors: ", bold: true, size: 22, color: COLORS.SECTION_M }),
                    new TextRun({ text: volume.eval_factors.join(", "), size: 22, color: COLORS.SECTION_M })
                ]
            }));
        }

        // Build sections
        const sections = volume.sections || [];
        if (sections.length > 0) {
            sections.forEach((section, secIndex) => {
                children.push(...buildSectionOutline(section, secIndex, volume, requirements, data));
            });
        } else {
            // No sections detected - create placeholder structure
            children.push(...buildPlaceholderSection(volume, requirements, data));
        }
    });

    // If no volumes, create a default structure
    if (volumes.length === 0) {
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun({ text: "PROPOSAL SECTIONS" })]
        }));
        children.push(new Paragraph({
            children: [new TextRun({
                text: "⚠ No volume structure detected. Review Section L for proposal organization requirements.",
                color: COLORS.SECTION_L, bold: true, size: 22
            })]
        }));
        children.push(...buildPlaceholderSection({name: "Technical Proposal"}, requirements, data));
    }

    return children;
}

/**
 * Build a single section with full annotations
 */
function buildSectionOutline(section, secIndex, volume, requirements, data) {
    const children = [];
    const sectionNum = `${section.id || (secIndex + 1)}`;

    // Section heading
    children.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [
            new TextRun({ text: `${sectionNum}. ${section.name || section.title || 'Section'}` })
        ]
    }));

    // Page allocation
    if (section.page_limit) {
        children.push(new Paragraph({
            spacing: { after: 100 },
            shading: { fill: "FFF2CC", type: ShadingType.CLEAR },
            children: [
                new TextRun({ text: "📄 PAGE ALLOCATION: ", bold: true, size: 20 }),
                new TextRun({ text: `${section.page_limit} pages`, bold: true, size: 20, color: COLORS.SECTION_L })
            ]
        }));
    }

    // === SECTION L REQUIREMENTS (RED) ===
    const sectionLReqs = (section.requirements || section.content_requirements || [])
        .filter(r => typeof r === 'string' && r.trim().length > 0);
    
    if (sectionLReqs.length > 0) {
        children.push(createAnnotationBlock(
            "SECTION L - INSTRUCTIONS",
            sectionLReqs,
            COLORS.SECTION_L,
            ANNOTATION_SHADING.L
        ));
    }

    // === SECTION M EVALUATION CRITERIA (BLUE) ===
    const evalCriteria = section.eval_criteria || [];
    if (evalCriteria.length > 0) {
        children.push(createAnnotationBlock(
            "SECTION M - EVALUATION CRITERIA",
            evalCriteria,
            COLORS.SECTION_M,
            ANNOTATION_SHADING.M
        ));
    }

    // === SECTION C/PWS REQUIREMENTS (PURPLE) ===
    // Smart semantic matching: find requirements relevant to this factor's topic
    const sectionId = section.id || '';
    const sectionName = section.name || section.title || '';
    
    // Extract factor number from section
    let factorNum = null;
    const idMatch = sectionId.match(/SEC-F(\d+)/i);
    const nameMatch = sectionName.match(/Factor\s*(\d+)/i);
    if (idMatch) factorNum = idMatch[1];
    else if (nameMatch) factorNum = nameMatch[1];
    
    // =========================================================================
    // FACTOR SEMANTIC PROFILES - What keywords indicate relevance to each factor
    // =========================================================================
    const factorProfiles = {
        // Factor 1: Experience / Corporate Experience
        "experience": {
            keywords: ["experience", "years", "demonstrated", "prior", "previous", 
                      "track record", "history", "performed", "completed", "successfully",
                      "relevant", "comparable", "similar", "contracts", "projects"],
            phrases: ["demonstrate experience", "relevant experience", "prior experience",
                     "similar in size", "similar in scope", "years of experience",
                     "have performed", "successfully completed"],
            weight: 1.0
        },
        // Factor 2: Program/Project Management  
        "management": {
            keywords: ["management", "project", "program", "plan", "planning",
                      "schedule", "milestone", "deliverable", "coordination",
                      "oversight", "control", "monitor", "track", "report",
                      "status", "progress", "timeline", "risk", "issue",
                      "mitigation", "communication", "stakeholder", "quality"],
            phrases: ["project management", "program management", "management approach",
                     "management plan", "project schedule", "work breakdown",
                     "risk management", "quality management", "status report",
                     "progress report", "milestone schedule"],
            weight: 1.0
        },
        // Factor 3: Technical Approach
        "technical": {
            keywords: ["technical", "approach", "methodology", "solution",
                      "design", "develop", "implement", "deploy", "architecture",
                      "system", "software", "hardware", "technology", "tool",
                      "platform", "framework", "process", "procedure", "method",
                      "technique", "innovation", "innovative", "integration",
                      "interface", "api", "data", "analysis", "research"],
            phrases: ["technical approach", "proposed approach", "methodology",
                     "solution design", "system architecture", "technical solution",
                     "implementation approach", "development methodology",
                     "best practices", "innovative approach"],
            weight: 1.0
        },
        // Factor 4: Key Personnel / Staffing
        "personnel": {
            keywords: ["personnel", "staff", "staffing", "team", "workforce",
                      "key", "principal", "lead", "manager", "director",
                      "resume", "cv", "biography", "qualifications", "credentials",
                      "education", "degree", "certification", "certified",
                      "pmp", "cissp", "skills", "expertise", "availability",
                      "commitment", "dedicated", "investigator", "specialist"],
            phrases: ["key personnel", "proposed personnel", "project manager",
                     "program manager", "technical lead", "subject matter expert",
                     "resumes", "qualifications", "years of experience",
                     "personnel commitment"],
            weight: 1.2
        },
        // Factor 5: Facilities / Equipment
        "facilities": {
            keywords: ["facility", "facilities", "equipment", "resource",
                      "infrastructure", "environment", "laboratory", "lab",
                      "office", "space", "building", "location", "hardware",
                      "server", "network", "computer", "tool", "software",
                      "license", "capability", "security", "clearance"],
            phrases: ["facilities and equipment", "laboratory facilities",
                     "office space", "computing resources", "infrastructure",
                     "security clearance", "security requirements",
                     "equipment availability"],
            weight: 1.0
        },
        // Factor 6: Past Performance
        "past_performance": {
            keywords: ["past performance", "performance", "reference",
                      "prior", "previous", "similar", "cpars", "ppirs",
                      "rating", "evaluation", "quality", "schedule", "cost",
                      "customer", "satisfaction", "successful", "on-time",
                      "relevance", "recency", "magnitude", "complexity"],
            phrases: ["past performance", "prior contracts", "previous contracts",
                     "contract references", "performance references",
                     "cpars ratings", "similar contracts", "performance record"],
            weight: 1.0
        }
    };
    
    // =========================================================================
    // FACTOR TYPE DETECTION - Determine what type this factor is
    // =========================================================================
    const detectFactorType = (name) => {
        const nameLower = name.toLowerCase();
        
        if (/experience|corporate\s+experience|relevant\s+experience/i.test(nameLower)) {
            return "experience";
        }
        if (/management|program|project|coordination/i.test(nameLower)) {
            return "management";
        }
        if (/technical|approach|methodology|solution|research/i.test(nameLower)) {
            return "technical";
        }
        if (/personnel|staff|team|key\s+personnel/i.test(nameLower)) {
            return "personnel";
        }
        if (/facilit|equipment|environment|infrastructure|security/i.test(nameLower)) {
            return "facilities";
        }
        if (/past\s+performance|performance\s+record|reference/i.test(nameLower)) {
            return "past_performance";
        }
        
        return null;
    };
    
    // =========================================================================
    // REQUIREMENT SCORING - Score how relevant a requirement is to this factor
    // =========================================================================
    const scoreRequirement = (reqText, factorType, factorNum) => {
        const textLower = (reqText || '').toLowerCase();
        let score = 0;
        let reasons = [];
        
        // Check for explicit factor mention (highest value)
        if (factorNum) {
            const factorPattern = new RegExp(`factor\\s*${factorNum}\\b`, 'i');
            if (factorPattern.test(textLower)) {
                score += 50;
                reasons.push(`Mentions Factor ${factorNum}`);
            }
        }
        
        // Get profile for this factor type
        const profile = factorProfiles[factorType];
        if (!profile) {
            return { score: 0, reason: "Unknown factor type" };
        }
        
        // Check phrase matches (high value)
        for (const phrase of profile.phrases) {
            if (textLower.includes(phrase.toLowerCase())) {
                score += 15 * profile.weight;
                reasons.push(`Phrase: "${phrase}"`);
            }
        }
        
        // Check keyword matches
        let keywordMatches = [];
        for (const keyword of profile.keywords) {
            if (textLower.includes(keyword.toLowerCase())) {
                score += 2 * profile.weight;
                keywordMatches.push(keyword);
            }
        }
        
        // Bonus for multiple keyword matches
        if (keywordMatches.length >= 3) {
            score += 10 * profile.weight;
            reasons.push(`Keywords: ${keywordMatches.slice(0, 3).join(", ")}`);
        } else if (keywordMatches.length > 0) {
            reasons.push(`Keywords: ${keywordMatches.slice(0, 2).join(", ")}`);
        }
        
        return {
            score: score,
            reason: reasons.length > 0 ? reasons.slice(0, 2).join("; ") : "Low relevance"
        };
    };
    
    // =========================================================================
    // MAIN MAPPING LOGIC - Find requirements for this factor
    // =========================================================================
    const factorType = detectFactorType(sectionName);
    let pwsReqs = [];
    
    if (factorType && requirements.length > 0) {
        // Score all requirements for this factor
        const scoredReqs = requirements
            .filter(r => {
                const text = r.text || r.full_text || '';
                return text.trim().length > 0;
            })
            .map(r => {
                const text = r.text || r.full_text || '';
                const result = scoreRequirement(text, factorType, factorNum);
                return {
                    ...r,
                    _score: result.score,
                    _reason: result.reason
                };
            })
            .filter(r => r._score > 5)  // Minimum relevance threshold
            .sort((a, b) => b._score - a._score);  // Sort by score descending
        
        // Take top 10 most relevant
        pwsReqs = scoredReqs.slice(0, 10);
    }
    
    // If we still have no requirements, try a simple keyword fallback
    if (pwsReqs.length === 0 && factorType) {
        const profile = factorProfiles[factorType];
        if (profile) {
            const topKeywords = profile.keywords.slice(0, 5);
            pwsReqs = requirements
                .filter(r => {
                    const text = (r.text || r.full_text || '').toLowerCase();
                    return topKeywords.some(kw => text.includes(kw.toLowerCase()));
                })
                .slice(0, 5);
        }
    }
    
    if (pwsReqs.length > 0) {
        children.push(createAnnotationBlock(
            "SECTION C/PWS - TECHNICAL REQUIREMENTS",
            pwsReqs.map(r => `[${r.req_id || 'REQ'}] ${r.text || r.full_text || ''}`),
            COLORS.SECTION_C,
            ANNOTATION_SHADING.C
        ));
    }

    // === WIN THEMES PLACEHOLDER (GREEN) ===
    children.push(createAnnotationBlock(
        "WIN THEMES & DISCRIMINATORS",
        [
            "[Enter discriminator/strength for this section]",
            "[Feature → Benefit → Proof structure]",
            "[Key message that aligns with customer hot buttons]"
        ],
        COLORS.WIN_THEME,
        ANNOTATION_SHADING.STRATEGY,
        true  // isPlaceholder
    ));

    // === PROOF POINTS PLACEHOLDER (ORANGE) ===
    children.push(createAnnotationBlock(
        "PROOF POINTS REQUIRED",
        [
            "[Past performance example needed]",
            "[Quantifiable metric to include (e.g., 99.9% uptime, 40% cost reduction)]",
            "[Certification or qualification to cite]"
        ],
        COLORS.PROOF_POINT,
        ANNOTATION_SHADING.PROOF,
        true
    ));

    // === GRAPHICS PLACEHOLDER ===
    children.push(createGraphicsPlaceholder());

    // === BOILERPLATE GUIDANCE ===
    children.push(createBoilerplateGuidance());

    // === CONTENT WRITING AREA ===
    children.push(new Paragraph({
        spacing: { before: 200, after: 100 },
        border: { bottom: { style: BorderStyle.DASHED, size: 1, color: COLORS.DARK_GRAY } },
        children: [
            new TextRun({ text: "CONTENT AREA", bold: true, size: 24, color: COLORS.DARK_GRAY })
        ]
    }));
    children.push(new Paragraph({
        spacing: { after: 100 },
        children: [
            new TextRun({ text: "[Writer: Begin proposal content here. Address ALL requirements above.]", size: 20, italics: true, color: COLORS.DARK_GRAY })
        ]
    }));
    // Add some blank space for writing
    for (let i = 0; i < 5; i++) {
        children.push(new Paragraph({ spacing: { after: 100 } }));
    }

    // Subsections
    const subsections = section.subsections || [];
    subsections.forEach((sub, subIndex) => {
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_3,
            children: [
                new TextRun({ text: `${sectionNum}.${subIndex + 1} ${sub.name || sub.title || 'Subsection'}` })
            ]
        }));
        children.push(new Paragraph({
            children: [
                new TextRun({ text: "[Add subsection annotations and content]", size: 20, italics: true, color: COLORS.DARK_GRAY })
            ]
        }));
    });

    return children;
}

/**
 * Create annotation block with colored label and requirements
 */
function createAnnotationBlock(label, items, labelColor, shading, isPlaceholder = false) {
    const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: labelColor };
    const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

    const contentParagraphs = items.map(item => 
        new Paragraph({
            numbering: { reference: "section-bullets", level: 0 },
            children: [new TextRun({
                text: item,
                size: 20,
                italics: isPlaceholder,
                color: isPlaceholder ? COLORS.DARK_GRAY : COLORS.BLACK
            })]
        })
    );

    return new Table({
        columnWidths: [9360],
        rows: [
            new TableRow({
                children: [
                    new TableCell({
                        borders: cellBorders,
                        width: { size: 9360, type: WidthType.DXA },
                        shading: shading,
                        children: [
                            new Paragraph({
                                spacing: { after: 80 },
                                children: [new TextRun({ text: label, bold: true, size: 18, color: labelColor, smallCaps: true })]
                            }),
                            ...contentParagraphs
                        ]
                    })
                ]
            })
        ]
    });
}

/**
 * Create graphics placeholder block
 */
function createGraphicsPlaceholder() {
    const tableBorder = { style: BorderStyle.DASHED, size: 1, color: COLORS.GRAPHIC };
    const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

    return new Table({
        columnWidths: [9360],
        rows: [
            new TableRow({
                children: [
                    new TableCell({
                        borders: cellBorders,
                        width: { size: 9360, type: WidthType.DXA },
                        shading: ANNOTATION_SHADING.GRAPHIC,
                        children: [
                            new Paragraph({
                                spacing: { after: 80 },
                                children: [new TextRun({ text: "PLANNED GRAPHICS", bold: true, size: 18, color: COLORS.GRAPHIC, smallCaps: true })]
                            }),
                            new Paragraph({
                                children: [new TextRun({ text: "Graphic 1: [Description of planned visual]", size: 20, italics: true })]
                            }),
                            new Paragraph({
                                indent: { left: 360 },
                                children: [new TextRun({
                                    text: "Action Caption: [Caption should convey key benefit, not just describe the figure]",
                                    size: 18, color: COLORS.DARK_GRAY
                                })]
                            }),
                            new Paragraph({
                                spacing: { before: 80 },
                                children: [new TextRun({ text: "Graphic 2: [Description]", size: 20, italics: true })]
                            }),
                            new Paragraph({
                                indent: { left: 360 },
                                children: [new TextRun({
                                    text: "Action Caption: [Caption text]",
                                    size: 18, color: COLORS.DARK_GRAY
                                })]
                            })
                        ]
                    })
                ]
            })
        ]
    });
}

/**
 * Create boilerplate guidance block
 */
function createBoilerplateGuidance() {
    return new Paragraph({
        spacing: { before: 120, after: 120 },
        shading: { fill: "F2F2F2", type: ShadingType.CLEAR },
        indent: { left: 180, right: 180 },
        children: [
            new TextRun({ text: "BOILERPLATE GUIDANCE: ", bold: true, size: 18, color: COLORS.BOILERPLATE, smallCaps: true }),
            new TextRun({ text: "[Specify: New content required / Adapt from [source] / Use standard boilerplate with tailoring]", size: 18, italics: true, color: COLORS.BOILERPLATE })
        ]
    });
}

/**
 * Build placeholder section when no structure detected
 */
function buildPlaceholderSection(volume, requirements, data) {
    const children = [];
    
    // Create generic technical sections
    const defaultSections = [
        { name: "Executive Summary", desc: "High-level overview demonstrating understanding and approach" },
        { name: "Technical Approach", desc: "Detailed methodology and solution design" },
        { name: "Management Approach", desc: "Project management, staffing, quality assurance" },
        { name: "Past Performance", desc: "Relevant experience and references" },
        { name: "Staffing Plan", desc: "Key personnel qualifications and org chart" }
    ];

    defaultSections.forEach((sec, idx) => {
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_2,
            children: [new TextRun({ text: `${idx + 1}. ${sec.name}` })]
        }));
        children.push(new Paragraph({
            spacing: { after: 120 },
            children: [
                new TextRun({ text: sec.desc, size: 20, italics: true, color: COLORS.DARK_GRAY })
            ]
        }));
        children.push(createAnnotationBlock(
            "REQUIREMENTS TO ADDRESS",
            ["[Review Section L and add specific requirements for this section]"],
            COLORS.SECTION_L,
            ANNOTATION_SHADING.L,
            true
        ));
        children.push(createAnnotationBlock(
            "WIN THEMES",
            ["[Add win themes and discriminators for this section]"],
            COLORS.WIN_THEME,
            ANNOTATION_SHADING.STRATEGY,
            true
        ));
        children.push(new Paragraph({ spacing: { after: 200 } }));
    });

    return children;
}

/**
 * Export document to buffer
 */
async function exportToBuffer(doc) {
    return await Packer.toBuffer(doc);
}

/**
 * Export document to file
 */
async function exportToFile(doc, filepath) {
    const buffer = await Packer.toBuffer(doc);
    fs.writeFileSync(filepath, buffer);
    return filepath;
}

// Main execution
async function main() {
    // Read input data from command line argument (JSON file path)
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.error("Usage: node annotated_outline_exporter.js <input.json> [output.docx]");
        process.exit(1);
    }

    const inputPath = args[0];
    const outputPath = args[1] || "annotated_outline.docx";

    try {
        const inputData = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
        const doc = generateAnnotatedOutline(inputData);
        await exportToFile(doc, outputPath);
        console.log(`Annotated outline generated: ${outputPath}`);
    } catch (error) {
        console.error("Error generating outline:", error);
        process.exit(1);
    }
}

// Export for use as module
module.exports = {
    generateAnnotatedOutline,
    exportToBuffer,
    exportToFile,
    COLORS
};

// Run if called directly
if (require.main === module) {
    main();
}
