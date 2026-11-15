"""Keep composite training IDs separate from the local GPT checkpoint clock."""


def training_state_id(record):
    return getattr(record, 'state_id', record.completed_steps)


def validate_graph_panel_identity(record, panel):
    """Validate immutable local cache identity and optional composite namespace."""
    sha = record.assert_verified()
    state = training_state_id(record)
    metadata = panel.metadata
    if metadata['state_sha256'] != sha or metadata['state_id'] != record.completed_steps:
        raise ValueError('teacher panel belongs to another verified local checkpoint')
    if 'training_state_id' in metadata and metadata['training_state_id'] != state:
        raise ValueError('teacher panel training-state namespace differs')
    if 'bank_seed' in metadata and metadata['bank_seed'] != getattr(record, 'bank_seed', None):
        raise ValueError('teacher panel bank seed differs')
    return state, sha
