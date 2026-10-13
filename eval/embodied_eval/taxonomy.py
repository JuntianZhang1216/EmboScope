"""Stable task names and the three current capability levels."""
GROUPS = {
    'L1': ('relative_distance', 'attribute_matching', 'cross_view_matching', 'frame_completion'),
    'L2': ('action_frame_matching', 'action_between_frames', 'step_frame_alignment',
           'action_sequence_description', 'adjacent_action', 'step_completion'),
    'L3': ('action_prediction_selection', 'progress_comparison', 'progress_ordering',
           'progress_anomaly_detection'),
}
