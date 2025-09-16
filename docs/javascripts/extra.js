// Initialize Mermaid diagrams
document.addEventListener('DOMContentLoaded', function() {
  // Check if mermaid is loaded
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      themeVariables: {
        primaryColor: '#1976d2',
        primaryTextColor: '#fff',
        primaryBorderColor: '#1976d2',
        lineColor: '#333',
        secondaryColor: '#f5f5f5',
        tertiaryColor: '#e3f2fd'
      },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      }
    });

    // Render any mermaid diagrams already on the page
    mermaid.contentLoaded();
  }
});
