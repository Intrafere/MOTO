import html2pdf from 'html2pdf.js';

/**
 * Download raw text content
 * @param {string} content - The text content
 * @param {string} filename - The filename (without extension)
 * @param {string} outline - Optional outline to prepend
 */
export const downloadRawText = (content, filename, outline = null) => {
  let fullContent = '';
  
  // Add outline if provided
  if (outline) {
    fullContent += 'OUTLINE\n';
    fullContent += '=' .repeat(80) + '\n\n';
    fullContent += outline + '\n\n';
    fullContent += '=' .repeat(80) + '\n\n';
  }
  
  // Add main content
  fullContent += content;
  
  const blob = new Blob([fullContent], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Download PDF from rendered HTML with metadata header
 * @param {HTMLElement} element - The DOM element to convert
 * @param {Object} metadata - Paper metadata {title, wordCount, date, models}
 * @param {string} filename - The filename (without extension)
 * @param {string} outline - Optional outline to prepend to the document
 */
export const downloadPDF = async (element, metadata, filename, outline = null) => {
  // Add logging for large documents
  const wordCount = metadata.wordCount || 0;
  if (wordCount > 40000) {
    console.warn(`Large document (${wordCount} words) - PDF generation may take 30-60 seconds...`);
  }
  
  // Clone element to avoid modifying original
  const clonedElement = element.cloneNode(true);
  
  // Set explicit white background and black text on cloned element
  clonedElement.style.setProperty('color', '#000', 'important');
  clonedElement.style.setProperty('background-color', '#ffffff', 'important');
  clonedElement.style.setProperty('background', '#ffffff', 'important');
  
  // Override all elements for proper light theme PDF rendering
  const allElements = clonedElement.querySelectorAll('*');
  allElements.forEach(el => {
    const tagName = el.tagName.toLowerCase();
    
    // Code blocks - dark text with light gray background
    if (tagName === 'code' || tagName === 'pre') {
      el.style.setProperty('color', '#222', 'important');
      el.style.setProperty('background-color', '#f5f5f5', 'important');
      el.style.setProperty('background', '#f5f5f5', 'important');
    } else {
      // Force all other text to black
      el.style.setProperty('color', '#000', 'important');
      // Force backgrounds to transparent (white container shows through)
      el.style.setProperty('background-color', 'transparent', 'important');
      el.style.setProperty('background', 'transparent', 'important');
    }
    
    // Handle SVG elements (KaTeX math rendering)
    if (tagName === 'svg') {
      // Force SVG to use black as current color
      el.style.setProperty('color', '#000', 'important');
    }
    
    // Force SVG text/tspan to black
    if (tagName === 'text' || tagName === 'tspan') {
      el.setAttribute('fill', '#000');
      el.style.setProperty('fill', '#000', 'important');
    }
    
    // Force SVG shapes to black (paths, lines, circles, rects)
    if (tagName === 'path' || tagName === 'line' || tagName === 'circle' || tagName === 'rect') {
      // Check computed fill - handle 'currentColor' and 'none' cases
      const computedFill = window.getComputedStyle(el).fill;
      if (computedFill === 'none') {
        // Path with no fill but possibly a stroke (like lines)
        el.setAttribute('stroke', '#000');
        el.style.setProperty('stroke', '#000', 'important');
      } else {
        // Filled shapes - set both fill and stroke to black
        el.setAttribute('fill', '#000');
        el.setAttribute('stroke', '#000');
        el.style.setProperty('fill', '#000', 'important');
        el.style.setProperty('stroke', '#000', 'important');
      }
    }
    
    // Handle border colors - convert to visible grey for PDF
    const computedStyle = window.getComputedStyle(el);
    const borderColor = computedStyle.borderColor;
    if (borderColor && borderColor !== 'rgba(0, 0, 0, 0)') {
      el.style.setProperty('border-color', '#ccc', 'important');
    }
  });
  
  // Create metadata header
  const header = document.createElement('div');
  header.style.cssText = 'margin-bottom: 20px; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; background-color: transparent;';
  header.innerHTML = `
    <h1 style="margin: 0 0 10px 0; color: #000; background: transparent;">${metadata.title || 'Untitled Paper'}</h1>
    <p style="margin: 5px 0; color: #333; background: transparent;">
      Word Count: ${metadata.wordCount?.toLocaleString() || 'N/A'} | 
      Generated: ${metadata.date || new Date().toLocaleDateString()}
    </p>
    ${metadata.models ? `<p style="margin: 5px 0; color: #333; font-size: 0.9em; background: transparent;">
      AI Models: ${metadata.models}
    </p>` : ''}
  `;
  
  // Prepend header to cloned content
  clonedElement.insertBefore(header, clonedElement.firstChild);
  
  // Add outline if provided
  if (outline) {
    const outlineSection = document.createElement('div');
    outlineSection.style.cssText = 'margin-bottom: 30px; padding: 15px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 4px;';
    outlineSection.innerHTML = `
      <h2 style="margin: 0 0 15px 0; color: #000; font-size: 1.3rem; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; background: transparent;">OUTLINE</h2>
      <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Georgia', 'Times New Roman', serif; font-size: 0.95rem; line-height: 1.6; color: #000; margin: 0; background: transparent;">${outline}</pre>
    `;
    clonedElement.insertBefore(outlineSection, clonedElement.children[1]); // Insert after header
  }
  
  // Wrap cloned element in a white container to ensure background is captured
  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'background-color: #ffffff !important; padding: 20px; min-height: 100%;';
  wrapper.appendChild(clonedElement);
  
  const options = {
    // Margins: [Top, Right, Bottom, Left] in mm
    // Bottom margin: 25mm for page numbers + ~10mm buffer for descenders to bleed into
    margin: [15, 15, 35, 15],
    filename: `${filename}.pdf`,
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { 
      scale: 2, 
      useCORS: true, 
      letterRendering: true,
      backgroundColor: '#ffffff'  // Force white background in canvas capture
    },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
  };
  
  try {
    console.log('Step 1/3: Preparing content...');
    // Use advanced API to add page numbers
    const worker = html2pdf().set(options).from(wrapper);
    
    console.log('Step 2/3: Rendering to canvas...');
    // Generate PDF with page numbers
    const pdf = await worker.toPdf().get('pdf');
    
    console.log('Step 3/3: Adding page numbers...');
    const totalPages = pdf.internal.getNumberOfPages();
    
    // Add page numbers to each page (bottom right)
    for (let i = 1; i <= totalPages; i++) {
      pdf.setPage(i);
      pdf.setFontSize(10);
      pdf.setTextColor(100);
      
      // Position: bottom right corner (210mm - 15mm margin = 195mm, 297mm - 10mm margin = 287mm)
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      pdf.text(`Page ${i} of ${totalPages}`, pageWidth - 15, pageHeight - 10, { align: 'right' });
    }
    
    console.log('PDF generation complete!');
    // Save the PDF
    pdf.save(`${filename}.pdf`);
  } catch (error) {
    console.error('PDF generation error:', error);
    if (wordCount > 40000) {
      throw new Error(`PDF generation failed for large document (${wordCount} words). Try "Download Raw" instead.`);
    }
    throw error;
  }
};

/**
 * Sanitize filename (remove special characters)
 */
export const sanitizeFilename = (filename) => {
  return filename
    .replace(/[^a-z0-9_\-\s]/gi, '')
    .replace(/\s+/g, '_')
    .substring(0, 100);
};


