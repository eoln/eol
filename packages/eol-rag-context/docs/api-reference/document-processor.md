# Document Processor Module

::: eol.rag_context.document_processor
    options:
      show_source: true
      show_bases: true
      show_root_heading: true
      heading_level: 2
      members_order: source
      show_signature_annotations: true
      separate_signature: true
      filters:
        - "!^_"  # Don't show private members
        - "^__init__"  # But do show __init__
      docstring_style: google
      docstring_options:
        ignore_init_summary: false
      merge_init_into_class: true
      show_if_no_docstring: false