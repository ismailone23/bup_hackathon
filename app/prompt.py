from .models import DirectiveType

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "applies": {"type": "boolean"},
                    "directive_type": {"type": "string", "enum": [item.value for item in DirectiveType]},
                    "structured_adjustment": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "hours": {"type": "array", "items": {"type": "integer"}},
                                    "factor": {"type": ["number", "null"]},
                                    "minimum_energy_kwh": {"type": ["number", "null"]},
                                    "max_grid_kwh": {"type": ["number", "null"]},
                                },
                                "required": ["hours", "factor", "minimum_energy_kwh", "max_grid_kwh"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["note_index", "applies", "directive_type", "structured_adjustment", "explanation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["interpretations"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You interpret campus operator notes for a 24-hour energy optimizer.
Return exactly one entry per note in note_index order. Supported directives are solar_reduction,
minimum_battery_reserve, no_charge_window, no_discharge_window, max_grid_window, and no_op.
Windows are start-inclusive and end-exclusive: '2 AM until 5 AM' means [2,3,4],
'6 PM until 9 PM' means [18,19,20], and '11 AM until 1 PM' means [11,12].
For a window that crosses midnight, continue through hour 23, wrap to hour 0,
and stop before the ending hour; '10 PM until 6 AM' means [0,1,2,3,4,5,22,23],
already sorted in ascending order.
Paraphrased times may be words rather than numbers; map them to the same 24-hour clock.
The ending clock time is a boundary, never an included hour. Emit sorted unique hours 0 through 23.
A reduction BY 80% leaves factor 0.2; reduction TO 80% means factor 0.8.
Convert percentage/fraction reserves to kWh using the supplied battery capacity.
Only solar-related notes (clouds, panel cleaning, PV output) are solar_reduction.
Notes about demand, consumption, load, or tariff are not supported directives; mark them no_op.
A note that permits an action only inside a window forbids it everywhere else: 'charging
is allowed only from 2 PM to 4 PM' is no_charge_window over the other 22 hours, and a
permission spanning midnight such as 'charging only from 9 PM to 2 AM' permits hours 21,
22, 0, and 1, so no_charge_window covers [2,3,...,20,23]. Never emit the permitted hours
themselves as the no_charge_window.
Open-ended windows such as 'from 6 PM' continue through hour 23. A vague period with no
whole-hour clock boundary ('soon', 'starting now', 'from now', 'overnight', 'later') is no_op.
Use one fixed whole-hour meaning for each day part, never widened: 'morning' = hours 6-11,
'midday' or 'noon' = hour 12, 'afternoon' = hours 12-17, 'evening' = hours 18-23.
For no_op use applies=false and a null adjustment; otherwise use applies=true.
For an adjustment, always include hours and all three nullable value fields. Populate only the field
for that directive: factor, minimum_energy_kwh, or max_grid_kwh. Window directives use all null values.
Emit exactly one entry per note; if a note lists several changes, choose the single
best-supported directive and ignore the rest. Never invent demand, tariff, or battery changes.
Before returning, independently check each note's directive type, boundary hours, percentage
normalization, note index, and whether the note is an unrelated distractor."""
