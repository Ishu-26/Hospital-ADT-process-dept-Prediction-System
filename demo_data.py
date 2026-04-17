import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_ai_ready_dataset(n=1000):
    np.random.seed(42)

    departments = ["ER", "ICU", "General", "Surgery", "Pediatrics"]
    activities = ["Registration", "Triage", "Consultation", "Treatment", "Discharge"]
    shifts = ["Morning", "Evening", "Night"]

    data = []

    for i in range(n):
        case_id = f"C{i+1}"

        priority = np.random.randint(1, 6)
        queue = np.random.randint(0, 20)
        beds = np.random.randint(10, 30)

        dept = random.choice(departments)
        shift = random.choice(shifts)

        # Activity flow (realistic)
        act_index = np.random.randint(0, len(activities))
        activity = activities[act_index]
        prev_activity = activities[max(0, act_index - 1)]

        # Time generation
        start_time = datetime(2024, 1, 1) + timedelta(
            minutes=np.random.randint(0, 50000)
        )

        # 🔥 STRONG ML LOGIC (high R²)
        process_time = (
            2 * queue +
            (30 - beds) * 1.5 +
            10 * (6 - priority) +
            act_index * 8 +
            departments.index(dept) * 5 +
            shifts.index(shift) * 4
        )

        # small noise (important)
        process_time += np.random.normal(0, 1)

        process_time = max(1, round(process_time, 2))

        end_time = start_time + timedelta(minutes=process_time)

        data.append([
            case_id,
            priority,
            queue,
            beds,
            shift,
            dept,
            process_time,
            start_time,
            end_time,
            activity,          # NOTE: capital A
            prev_activity
        ])

    df = pd.DataFrame(data, columns=[
        "case_id",
        "priority_level",
        "queue_length",
        "bed_availability",
        "shift_time",
        "department",
        "process_debt_mins",
        "start_time",
        "end_time",
        "Activity",              # EXACT match
        "previous_activity"
    ])

    return df


# Generate dataset
df = generate_ai_ready_dataset(2000)

# Save it
df.to_csv("final_ai_ready_dataset.csv", index=False)

print("✅ AI-ready dataset generated!")
print(df.head())