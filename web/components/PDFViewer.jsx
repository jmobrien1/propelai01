/**
 * PropelAI Trust Gate PDF Viewer
 * Interactive PDF viewer with click-to-verify highlighting
 *
 * Usage:
 *   <PDFViewer
 *     pdfUrl="https://..."
 *     highlights={[{page: 1, x: 100, y: 200, width: 300, height: 20}]}
 *     initialPage={1}
 *     onTextSelect={(text, bbox) => console.log(text)}
 *   />
 */

// This component is designed to be embedded in the main index.html
// It uses react-pdf for rendering and custom overlay for highlights

const PDFViewerStyles = `
  .pdf-viewer-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-secondary, #12121a);
    border-radius: 8px;
    overflow: hidden;
  }

  .pdf-viewer-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg-tertiary, #1a1a24);
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }

  .pdf-viewer-toolbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .pdf-viewer-toolbar-center {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .pdf-viewer-toolbar-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .pdf-page-input {
    width: 50px;
    padding: 4px 8px;
    background: var(--bg-primary, #0a0a0f);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px;
    color: white;
    text-align: center;
  }

  .pdf-viewer-btn {
    padding: 6px 12px;
    background: transparent;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px;
    color: white;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
  }

  .pdf-viewer-btn:hover {
    background: rgba(255,255,255,0.1);
  }

  .pdf-viewer-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .pdf-viewer-btn-primary {
    background: var(--accent-blue, #4f8cff);
    border-color: var(--accent-blue, #4f8cff);
  }

  .pdf-viewer-btn-primary:hover {
    background: var(--accent-blue-hover, #3a7ae8);
  }

  .pdf-viewer-content {
    flex: 1;
    overflow: auto;
    display: flex;
    justify-content: center;
    padding: 20px;
    position: relative;
  }

  .pdf-page-container {
    position: relative;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }

  .pdf-page-canvas {
    display: block;
  }

  .pdf-highlight-overlay {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
  }

  .pdf-highlight {
    position: absolute;
    background: rgba(255, 255, 0, 0.3);
    border: 2px solid rgba(255, 200, 0, 0.8);
    border-radius: 2px;
    animation: highlight-pulse 2s ease-in-out infinite;
  }

  @keyframes highlight-pulse {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
  }

  .pdf-highlight-source {
    background: rgba(79, 140, 255, 0.3);
    border-color: rgba(79, 140, 255, 0.8);
  }

  .pdf-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: rgba(255,255,255,0.6);
  }

  .pdf-error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--accent-red, #f87171);
    text-align: center;
    padding: 20px;
  }

  .pdf-zoom-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .pdf-zoom-label {
    font-size: 12px;
    color: rgba(255,255,255,0.6);
    min-width: 50px;
    text-align: center;
  }

  .verification-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
  }

  .verification-badge.verified {
    background: rgba(52, 211, 153, 0.2);
    color: #34d399;
  }

  .verification-badge.unverified {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
  }

  .verification-badge.not-found {
    background: rgba(248, 113, 113, 0.2);
    color: #f87171;
  }
`;

// Inject styles
if (typeof document !== 'undefined') {
  const styleEl = document.createElement('style');
  styleEl.textContent = PDFViewerStyles;
  document.head.appendChild(styleEl);
}

/**
 * PDF Viewer Component
 */
function PDFViewer({
  pdfUrl,
  highlights = [],
  initialPage = 1,
  onTextSelect,
  onPageChange,
  verificationStatus,
  documentName,
}) {
  const [currentPage, setCurrentPage] = React.useState(initialPage);
  const [totalPages, setTotalPages] = React.useState(0);
  const [scale, setScale] = React.useState(1.0);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [pdfDoc, setPdfDoc] = React.useState(null);

  const canvasRef = React.useRef(null);
  const containerRef = React.useRef(null);

  // Load PDF.js library dynamically
  React.useEffect(() => {
    if (!window.pdfjsLib) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
      script.onload = () => {
        window.pdfjsLib.GlobalWorkerOptions.workerSrc =
          'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        loadPDF();
      };
      document.head.appendChild(script);
    } else {
      loadPDF();
    }
  }, [pdfUrl]);

  // Load PDF document
  async function loadPDF() {
    if (!pdfUrl || !window.pdfjsLib) return;

    setLoading(true);
    setError(null);

    try {
      const loadingTask = window.pdfjsLib.getDocument(pdfUrl);
      const pdf = await loadingTask.promise;
      setPdfDoc(pdf);
      setTotalPages(pdf.numPages);
      setCurrentPage(Math.min(initialPage, pdf.numPages));
    } catch (err) {
      setError(`Failed to load PDF: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  // Render current page
  React.useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return;

    async function renderPage() {
      try {
        const page = await pdfDoc.getPage(currentPage);
        const viewport = page.getViewport({ scale });

        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');

        canvas.width = viewport.width;
        canvas.height = viewport.height;

        await page.render({
          canvasContext: context,
          viewport: viewport,
        }).promise;

        if (onPageChange) {
          onPageChange(currentPage);
        }
      } catch (err) {
        console.error('Error rendering page:', err);
      }
    }

    renderPage();
  }, [pdfDoc, currentPage, scale]);

  // Navigation handlers
  function goToPage(page) {
    const newPage = Math.max(1, Math.min(page, totalPages));
    setCurrentPage(newPage);
  }

  function handlePageInput(e) {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      goToPage(value);
    }
  }

  // Zoom handlers
  function zoomIn() {
    setScale(s => Math.min(s + 0.25, 3.0));
  }

  function zoomOut() {
    setScale(s => Math.max(s - 0.25, 0.5));
  }

  function resetZoom() {
    setScale(1.0);
  }

  // Get highlights for current page
  const pageHighlights = highlights.filter(h => h.page === currentPage);

  // Render verification badge
  function renderVerificationBadge() {
    if (!verificationStatus) return null;

    const { verified, confidence } = verificationStatus;

    if (verified) {
      return (
        <span className="verification-badge verified">
          ✓ Verified ({Math.round(confidence * 100)}%)
        </span>
      );
    } else if (confidence > 0) {
      return (
        <span className="verification-badge unverified">
          ⚠ Partial Match ({Math.round(confidence * 100)}%)
        </span>
      );
    } else {
      return (
        <span className="verification-badge not-found">
          ✗ Not Found
        </span>
      );
    }
  }

  if (loading) {
    return (
      <div className="pdf-viewer-container">
        <div className="pdf-loading">
          Loading PDF...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="pdf-viewer-container">
        <div className="pdf-error">
          <p>{error}</p>
          <button className="pdf-viewer-btn" onClick={loadPDF}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="pdf-viewer-container" ref={containerRef}>
      {/* Toolbar */}
      <div className="pdf-viewer-toolbar">
        <div className="pdf-viewer-toolbar-left">
          <span style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)' }}>
            {documentName || 'Document'}
          </span>
          {renderVerificationBadge()}
        </div>

        <div className="pdf-viewer-toolbar-center">
          <button
            className="pdf-viewer-btn"
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage <= 1}
          >
            ←
          </button>
          <span style={{ color: 'rgba(255,255,255,0.8)', fontSize: '14px' }}>
            <input
              type="number"
              className="pdf-page-input"
              value={currentPage}
              onChange={handlePageInput}
              min={1}
              max={totalPages}
            />
            {' / '}{totalPages}
          </span>
          <button
            className="pdf-viewer-btn"
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage >= totalPages}
          >
            →
          </button>
        </div>

        <div className="pdf-viewer-toolbar-right">
          <div className="pdf-zoom-controls">
            <button className="pdf-viewer-btn" onClick={zoomOut}>−</button>
            <span className="pdf-zoom-label">{Math.round(scale * 100)}%</span>
            <button className="pdf-viewer-btn" onClick={zoomIn}>+</button>
            <button className="pdf-viewer-btn" onClick={resetZoom}>Reset</button>
          </div>
        </div>
      </div>

      {/* PDF Content */}
      <div className="pdf-viewer-content">
        <div className="pdf-page-container">
          <canvas ref={canvasRef} className="pdf-page-canvas" />

          {/* Highlight Overlay */}
          <svg
            className="pdf-highlight-overlay"
            style={{
              width: canvasRef.current?.width || 0,
              height: canvasRef.current?.height || 0,
            }}
          >
            {pageHighlights.map((highlight, idx) => (
              <rect
                key={idx}
                x={highlight.x * scale}
                y={highlight.y * scale}
                width={highlight.width * scale}
                height={highlight.height * scale}
                className={`pdf-highlight ${highlight.isSource ? 'pdf-highlight-source' : ''}`}
                style={{
                  fill: highlight.color || 'rgba(255, 255, 0, 0.3)',
                  stroke: highlight.color || 'rgba(255, 200, 0, 0.8)',
                }}
              />
            ))}
          </svg>
        </div>
      </div>
    </div>
  );
}

/**
 * Trust Gate Panel Component
 * Split-panel view with requirements table and PDF viewer
 */
function TrustGatePanel({
  requirement,
  rfpId,
  onClose,
}) {
  const [loading, setLoading] = React.useState(true);
  const [verificationData, setVerificationData] = React.useState(null);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    if (requirement) {
      loadVerificationData();
    }
  }, [requirement]);

  async function loadVerificationData() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `/api/trust-gate/rfp/${rfpId}/requirements/${requirement.id}/verify`
      );

      if (!response.ok) {
        throw new Error('Failed to load verification data');
      }

      const data = await response.json();
      setVerificationData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="trust-gate-panel">
        <div className="trust-gate-loading">
          Loading verification data...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="trust-gate-panel">
        <div className="trust-gate-error">
          <p>{error}</p>
          <button onClick={loadVerificationData}>Retry</button>
        </div>
      </div>
    );
  }

  const { source, viewer_data } = verificationData || {};

  return (
    <div className="trust-gate-panel">
      <div className="trust-gate-header">
        <h3>Source Verification</h3>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>

      <div className="trust-gate-content">
        {/* Requirement Info */}
        <div className="trust-gate-requirement">
          <h4>Requirement</h4>
          <p>{requirement.text}</p>
          <div className="requirement-meta">
            <span>Section: {requirement.section}</span>
            <span>Page: {source?.page || requirement.source_page || 'N/A'}</span>
            <span>Confidence: {Math.round((source?.confidence || 0) * 100)}%</span>
          </div>
        </div>

        {/* PDF Viewer */}
        {viewer_data && (
          <div className="trust-gate-viewer">
            <PDFViewer
              pdfUrl={viewer_data.pdf_url}
              highlights={viewer_data.highlights.map(h => ({
                ...h,
                isSource: true,
              }))}
              initialPage={viewer_data.initial_page}
              documentName={viewer_data.document_name}
              verificationStatus={{
                verified: verificationData.verified,
                confidence: source?.confidence || 0,
              }}
            />
          </div>
        )}

        {/* Context */}
        {source?.context_before || source?.context_after ? (
          <div className="trust-gate-context">
            <h4>Context</h4>
            {source.context_before && (
              <div className="context-before">
                <span className="context-label">Before:</span>
                <p>{source.context_before}</p>
              </div>
            )}
            <div className="context-match">
              <span className="context-label">Match:</span>
              <p>{source.matched_text}</p>
            </div>
            {source.context_after && (
              <div className="context-after">
                <span className="context-label">After:</span>
                <p>{source.context_after}</p>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}

// Export for use in main app
if (typeof window !== 'undefined') {
  window.PDFViewer = PDFViewer;
  window.TrustGatePanel = TrustGatePanel;
}
