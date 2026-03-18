from .curl_form_feedback import analyze_rep as analyze_curl_rep
from .shoulder_press_form_feedback import analyze_rep as analyze_shoulder_press_rep
from .squat_form_feedback import analyze_rep as analyze_squat_rep

# Add your own exercise feedback test function here.
# Example:
# from .deadlift_form_feedback import analyze_rep as analyze_deadlift_rep
# EXERCISE_ANALYZERS["deadlift"] = analyze_deadlift_rep
EXERCISE_ANALYZERS = {
    "curl": analyze_curl_rep,
    "shoulder_press": analyze_shoulder_press_rep,
    "squat": analyze_squat_rep,
}


def get_exercise_analyzer(exercise_name):
    return EXERCISE_ANALYZERS.get(exercise_name)
